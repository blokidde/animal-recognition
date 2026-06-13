from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO


@dataclass
class StableLabel:
    class_id: int
    class_name: str
    confidence: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLO on a video and keep the highest-confidence class per tracked animal."
    )
    parser.add_argument("--model", required=True, type=Path, help="YOLO .pt model path")
    parser.add_argument("--source", required=True, type=Path, help="Input video path")
    parser.add_argument("--save", required=True, type=Path, help="Output video path")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.001, help="Detection threshold. Use a very low value to inspect almost all model proposals.")
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--device", default="0")
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument(
        "--min-update-conf",
        type=float,
        default=0.0,
        help="Only update a track's remembered label when the current detection confidence is at least this value. The remembered label still only changes when confidence improves.",
    )
    parser.add_argument(
        "--line-thickness",
        type=int,
        default=2,
        help="Bounding box line thickness.",
    )
    return parser.parse_args()


def get_class_name(names: dict[int, str] | list[str], class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def track_color(track_id: int) -> tuple[int, int, int]:
    # BGR color, deterministic per track id.
    return (
        60 + (track_id * 37) % 170,
        60 + (track_id * 17) % 170,
        60 + (track_id * 29) % 170,
    )


def draw_label(
    frame: Any,
    box: tuple[int, int, int, int],
    label: str,
    color: tuple[int, int, int],
    line_thickness: int,
) -> None:
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, line_thickness)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    text_thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)
    y_text_top = max(0, y1 - text_height - baseline - 6)
    cv2.rectangle(
        frame,
        (x1, y_text_top),
        (x1 + text_width + 8, y_text_top + text_height + baseline + 6),
        color,
        -1,
    )
    cv2.putText(
        frame,
        label,
        (x1 + 4, y_text_top + text_height + 3),
        font,
        font_scale,
        (255, 255, 255),
        text_thickness,
        cv2.LINE_AA,
    )


def make_output_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix or ".mp4"
    for index in range(1, 1000):
        candidate = path.with_name(f"{stem}_{index}{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not create a unique output path for {path}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"Model does not exist: {args.model}")
    if not args.source.exists():
        raise FileNotFoundError(f"Source video does not exist: {args.source}")

    output_path = make_output_path(args.save)
    model = YOLO(str(args.model))

    cap = cv2.VideoCapture(str(args.source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open output video writer: {output_path}")

    stable_labels: dict[int, StableLabel] = {}
    frame_index = 0
    logging.info("Processing %s frames from %s", frame_count or "unknown", args.source)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1

            results = model.track(
                frame,
                persist=True,
                tracker=args.tracker,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                verbose=False,
            )
            result = results[0]
            boxes = result.boxes

            if boxes is not None and boxes.id is not None:
                xyxy = boxes.xyxy.cpu().numpy()
                track_ids = boxes.id.cpu().numpy().astype(int)
                class_ids = boxes.cls.cpu().numpy().astype(int)
                confidences = boxes.conf.cpu().numpy()

                for coords, track_id, class_id, confidence in zip(xyxy, track_ids, class_ids, confidences):
                    current_name = get_class_name(model.names, int(class_id))
                    confidence = float(confidence)
                    previous = stable_labels.get(int(track_id))
                    if confidence >= args.min_update_conf and (
                        previous is None or confidence > previous.confidence
                    ):
                        stable_labels[int(track_id)] = StableLabel(
                            class_id=int(class_id),
                            class_name=current_name,
                            confidence=confidence,
                        )

                    stable = stable_labels.get(int(track_id))
                    if stable is None:
                        stable = StableLabel(int(class_id), current_name, confidence)

                    x1, y1, x2, y2 = [int(round(value)) for value in coords]
                    label = f"{stable.class_name} {stable.confidence:.2f} id:{track_id}"
                    draw_label(
                        frame,
                        (x1, y1, x2, y2),
                        label,
                        track_color(int(track_id)),
                        args.line_thickness,
                    )

            writer.write(frame)
            if frame_index % 100 == 0:
                logging.info("Processed %s/%s frames", frame_index, frame_count or "?")
    finally:
        writer.release()
        cap.release()

    logging.info("Saved stable-label video to %s", output_path)


if __name__ == "__main__":
    main()

