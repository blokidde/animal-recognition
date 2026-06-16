from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any
try:
    from windows_platform_patch import patch_windows_platform_wmi
except ImportError:
    from classifier.windows_platform_patch import patch_windows_platform_wmi

patch_windows_platform_wmi()

import cv2
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import models, transforms
from ultralytics import YOLO

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class CropPrediction:
    class_id: int
    class_name: str
    confidence: float
    probabilities: list[float]


@dataclass
class TrackState:
    score_sum: list[float]
    weight_sum: float
    votes: int
    best_detector_conf: float

    def update(self, prediction: CropPrediction, detector_confidence: float) -> None:
        weight = max(0.01, float(detector_confidence))
        if not self.score_sum:
            self.score_sum = [0.0 for _ in prediction.probabilities]
        for index, probability in enumerate(prediction.probabilities):
            self.score_sum[index] += probability * weight
        self.weight_sum += weight
        self.votes += 1
        self.best_detector_conf = max(self.best_detector_conf, float(detector_confidence))

    def best(self, class_names: list[str]) -> tuple[int, str, float, float]:
        if self.weight_sum <= 0 or not self.score_sum:
            return 0, class_names[0], 0.0, 0.0
        averaged = [score / self.weight_sum for score in self.score_sum]
        class_id = max(range(len(averaged)), key=lambda index: averaged[index])
        cls_conf = float(averaged[class_id])
        combined_conf = cls_conf * min(1.0, self.best_detector_conf)
        return class_id, class_names[class_id], cls_conf, combined_conf


class Letterbox:
    def __init__(self, size: int, fill: tuple[int, int, int] = (114, 114, 114)) -> None:
        self.size = size
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        width, height = image.size
        if width <= 0 or height <= 0:
            return Image.new("RGB", (self.size, self.size), self.fill)
        scale = min(self.size / width, self.size / height)
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        image = image.resize((new_width, new_height), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (self.size, self.size), self.fill)
        canvas.paste(image, ((self.size - new_width) // 2, (self.size - new_height) // 2))
        return canvas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO bboxes plus a custom crop classifier on video.")
    parser.add_argument("--detector", default="yolo26n.pt", help="YOLO detector model for bbox proposals")
    parser.add_argument("--classifier", required=True, type=Path, help="Classifier checkpoint best.pt/last.pt")
    parser.add_argument("--source", required=True, type=Path, help="Input video")
    parser.add_argument("--save", required=True, type=Path, help="Output video")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--det-conf", type=float, default=0.25)
    parser.add_argument("--det-iou", type=float, default=0.5)
    parser.add_argument("--cls-conf", type=float, default=0.65, help="Minimum averaged classifier confidence to draw")
    parser.add_argument("--combined-conf", type=float, default=0.30, help="Minimum cls_conf * best_detector_conf to draw")
    parser.add_argument("--min-track-votes", type=int, default=2, help="Minimum classifier updates before drawing a track")
    parser.add_argument("--device", default="0")
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument("--crop-margin", type=float, default=0.18)
    parser.add_argument("--classify-every", type=int, default=3, help="Run classifier every N frames per track")
    parser.add_argument("--max-crops-per-frame", type=int, default=64)
    parser.add_argument("--allowed-detector-classes", nargs="*", default=["bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"])
    return parser.parse_args()


def create_classifier(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_classifier(path: Path, device: torch.device) -> tuple[nn.Module, list[str], int]:
    checkpoint = torch.load(path, map_location=device)
    display_names = list(checkpoint.get("display_names") or checkpoint.get("class_names"))
    image_size = int(checkpoint.get("image_size", 384))
    model = create_classifier(len(display_names))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, display_names, image_size


def expand_box(box: tuple[int, int, int, int], margin: float, width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    pad_x = int(round(box_w * margin))
    pad_y = int(round(box_h * margin))
    return max(0, x1 - pad_x), max(0, y1 - pad_y), min(width, x2 + pad_x), min(height, y2 + pad_y)


def classify_crop(
    frame: Any,
    box: tuple[int, int, int, int],
    model: nn.Module,
    transform: transforms.Compose,
    class_names: list[str],
    device: torch.device,
) -> CropPrediction:
    x1, y1, x2, y2 = box
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return CropPrediction(0, class_names[0], 0.0, [1.0] + [0.0 for _ in class_names[1:]])
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = model(tensor)
        probs_tensor = torch.softmax(logits, dim=1)[0]
        confidence, class_id = torch.max(probs_tensor, dim=0)
    probabilities = [float(value) for value in probs_tensor.detach().cpu().tolist()]
    idx = int(class_id.item())
    return CropPrediction(idx, class_names[idx], float(confidence.item()), probabilities)


def track_color(track_id: int) -> tuple[int, int, int]:
    track_id = int(track_id)
    return (
        int(60 + (track_id * 37) % 170),
        int(60 + (track_id * 17) % 170),
        int(60 + (track_id * 29) % 170),
    )


def draw_label(frame: Any, box: tuple[int, int, int, int], label: str, color: tuple[int, int, int]) -> None:
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    top = max(0, y1 - th - baseline - 6)
    cv2.rectangle(frame, (x1, top), (x1 + tw + 8, top + th + baseline + 6), color, -1)
    cv2.putText(frame, label, (x1 + 4, top + th + 3), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)


def make_output_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        return path
    for index in range(1, 1000):
        candidate = path.with_name(f"{path.stem}_{index}{path.suffix or '.mp4'}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not create unique path for {path}")


def detector_class_name(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    return str(names[class_id]) if 0 <= class_id < len(names) else str(class_id)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    if not args.source.exists():
        raise FileNotFoundError(f"Source video does not exist: {args.source}")
    if not args.classifier.exists():
        raise FileNotFoundError(f"Classifier checkpoint does not exist: {args.classifier}")

    allowed_detector_classes = set(args.allowed_detector_classes or [])
    device = torch.device(f"cuda:{args.device}" if args.device != "cpu" and torch.cuda.is_available() else "cpu")
    detector = YOLO(args.detector)
    classifier, class_names, image_size = load_classifier(args.classifier, device)
    transform = transforms.Compose([Letterbox(image_size), transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

    cap = cv2.VideoCapture(str(args.source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    output_path = make_output_path(args.save)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open output video writer: {output_path}")

    track_states: dict[int, TrackState] = {}
    last_classified_frame: dict[int, int] = {}
    frame_index = 0
    logging.info("Processing %s frames from %s", total_frames or "unknown", args.source)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1
            results = detector.track(
                frame,
                persist=True,
                tracker=args.tracker,
                imgsz=args.imgsz,
                conf=args.det_conf,
                iou=args.det_iou,
                device=args.device,
                verbose=False,
            )
            boxes = results[0].boxes
            if boxes is not None and boxes.id is not None:
                xyxy = boxes.xyxy.cpu().numpy()
                track_ids = boxes.id.cpu().numpy().astype(int)
                det_confs = boxes.conf.cpu().numpy()
                det_classes = boxes.cls.cpu().numpy().astype(int)
                handled = 0
                for coords, track_id, det_conf, det_class_id in zip(xyxy, track_ids, det_confs, det_classes):
                    if handled >= args.max_crops_per_frame:
                        break
                    det_class = detector_class_name(detector.names, int(det_class_id))
                    if allowed_detector_classes and det_class not in allowed_detector_classes:
                        continue
                    raw_box = tuple(int(round(v)) for v in coords)
                    crop_box = expand_box(raw_box, args.crop_margin, width, height)
                    should_classify = (
                        int(track_id) not in track_states
                        or frame_index - last_classified_frame.get(int(track_id), -10**9) >= max(1, args.classify_every)
                    )
                    if should_classify:
                        prediction = classify_crop(frame, crop_box, classifier, transform, class_names, device)
                        state = track_states.setdefault(int(track_id), TrackState([], 0.0, 0, 0.0))
                        state.update(prediction, float(det_conf))
                        last_classified_frame[int(track_id)] = frame_index
                    state = track_states.get(int(track_id))
                    if state is not None and state.votes >= args.min_track_votes:
                        _, class_name, cls_conf, combined_conf = state.best(class_names)
                        if cls_conf >= args.cls_conf and combined_conf >= args.combined_conf:
                            label = f"{class_name} {cls_conf:.2f}/{combined_conf:.2f} id:{int(track_id)}"
                            draw_label(frame, raw_box, label, track_color(int(track_id)))
                    handled += 1
            writer.write(frame)
            if frame_index % 100 == 0:
                logging.info("Processed %s/%s frames", frame_index, total_frames or "?")
    finally:
        writer.release()
        cap.release()

    metadata = {
        "detector": args.detector,
        "classifier": str(args.classifier),
        "source": str(args.source),
        "output": str(output_path),
        "classify_every": args.classify_every,
        "crop_margin": args.crop_margin,
        "det_conf": args.det_conf,
        "det_iou": args.det_iou,
        "cls_conf": args.cls_conf,
        "combined_conf": args.combined_conf,
        "min_track_votes": args.min_track_votes,
        "allowed_detector_classes": sorted(allowed_detector_classes),
    }
    output_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logging.info("Saved detector+classifier video to %s", output_path)


if __name__ == "__main__":
    main()

