from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    from windows_platform_patch import patch_windows_platform_wmi
except ImportError:
    from classifier.windows_platform_patch import patch_windows_platform_wmi

patch_windows_platform_wmi()

import yaml
from PIL import Image, ImageOps
from ultralytics import YOLO

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEFAULT_CONFIG = Path(__file__).resolve().parent / "params_yolo_detector_crops.yaml"


@dataclass(frozen=True)
class CropRow:
    species: str
    source_path: str
    crop_path: str
    detector_class: str
    detector_confidence: float
    x1: int
    y1: int
    x2: int
    y2: int
    crop_index: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create runtime-like classifier crops with a YOLO bbox detector.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config does not exist: {path}")
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return data


def resolve_path(value: str | Path) -> Path:
    return Path(value).resolve()


def safe_stem(value: str, max_length: int = 90) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned[:max_length] or "image"


def list_images(source_root: Path) -> list[Path]:
    paths: list[Path] = []
    for species_dir in sorted(path for path in source_root.iterdir() if path.is_dir()):
        for path in sorted(species_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                paths.append(path)
    return paths


def iter_chunks(items: list[Path], chunk_size: int) -> Iterable[list[Path]]:
    chunk_size = max(1, chunk_size)
    for start in range(0, len(items), chunk_size):
        yield items[start:start + chunk_size]


def expand_box(x1: float, y1: float, x2: float, y2: float, margin: float, width: int, height: int) -> tuple[int, int, int, int]:
    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)
    pad_x = box_w * margin
    pad_y = box_h * margin
    return (
        max(0, int(round(x1 - pad_x))),
        max(0, int(round(y1 - pad_y))),
        min(width, int(round(x2 + pad_x))),
        min(height, int(round(y2 + pad_y))),
    )


def crop_image(source_path: Path, box: tuple[int, int, int, int], destination: Path, quality: int) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        crop = image.crop(box)
        crop.save(destination, format="JPEG", quality=quality, optimize=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(parse_args().config)

    source_root = resolve_path(config["source_root"])
    output_root = resolve_path(config["output_root"])
    crops_root = output_root / "crops"
    metadata_dir = output_root / "_metadata"
    detector_name = str(config.get("detector", "yolo26n.pt"))
    conf = float(config.get("conf", 0.20))
    iou = float(config.get("iou", 0.5))
    imgsz = int(config.get("imgsz", 640))
    batch = int(config.get("batch", 32))
    chunk_size = int(config.get("chunk_size", max(batch * 4, batch)))
    device = str(config.get("device", "0"))
    margin = float(config.get("crop_margin", 0.15))
    max_crops_per_image = int(config.get("max_crops_per_image", 3))
    min_area_ratio = float(config.get("min_area_ratio", 0.001))
    jpeg_quality = int(config.get("jpeg_quality", 95))
    max_images = config.get("max_images")
    regenerate = bool(config.get("regenerate", False))
    allowed_detector_classes = set(str(name) for name in config.get("allowed_detector_classes", []))

    if not source_root.exists():
        raise FileNotFoundError(f"Source root does not exist: {source_root}")
    if regenerate and output_root.exists():
        shutil.rmtree(output_root)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(source_root)
    if max_images is not None:
        images = images[: int(max_images)]
    if not images:
        raise RuntimeError(f"No images found under {source_root}")

    model = YOLO(detector_name)
    names = model.names
    rows: list[CropRow] = []
    no_detection_count = 0
    total_crops = 0

    logging.info(
        "Running %s on %s images from %s in chunks of %s",
        detector_name,
        len(images),
        source_root,
        chunk_size,
    )
    for processed, source_path in enumerate(images, start=1):
        if processed == 1 or processed % 250 == 0:
            logging.info("Predicting image %s/%s", processed, len(images))
        results = model.predict(
            source=str(source_path),
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device=device,
            batch=batch,
            stream=False,
            verbose=False,
        )
        if not results:
            no_detection_count += 1
            continue
        result = results[0]
        species = source_path.parent.name
        with Image.open(source_path) as image:
            image_width, image_height = image.size
        candidates: list[tuple[float, str, tuple[int, int, int, int]]] = []

        if result.boxes is not None and len(result.boxes) > 0:
            xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            for coords, det_conf, cls_id in zip(xyxy, confs, classes):
                detector_class = str(names.get(int(cls_id), cls_id)) if isinstance(names, dict) else str(names[int(cls_id)])
                if allowed_detector_classes and detector_class not in allowed_detector_classes:
                    continue
                x1, y1, x2, y2 = [float(value) for value in coords]
                area_ratio = max(0.0, (x2 - x1) * (y2 - y1)) / max(1, image_width * image_height)
                if area_ratio < min_area_ratio:
                    continue
                box = expand_box(x1, y1, x2, y2, margin, image_width, image_height)
                candidates.append((float(det_conf), detector_class, box))

        candidates.sort(key=lambda item: item[0], reverse=True)
        if not candidates:
            no_detection_count += 1
        for crop_index, (det_conf, detector_class, box) in enumerate(candidates[:max_crops_per_image], start=1):
            crop_name = f"{safe_stem(source_path.stem)}__det{crop_index:02d}__{det_conf:.3f}.jpg"
            crop_path = crops_root / species / crop_name
            crop_image(source_path, box, crop_path, jpeg_quality)
            rows.append(
                CropRow(
                    species=species,
                    source_path=str(source_path),
                    crop_path=str(crop_path),
                    detector_class=detector_class,
                    detector_confidence=det_conf,
                    x1=box[0],
                    y1=box[1],
                    x2=box[2],
                    y2=box[3],
                    crop_index=crop_index,
                )
            )
            total_crops += 1

        if processed % 500 == 0:
            logging.info("Processed %s/%s images, crops=%s, no_detection=%s", processed, len(images), total_crops, no_detection_count)

    manifest_csv = metadata_dir / "yolo_detector_crops.csv"
    with manifest_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(CropRow.__dataclass_fields__.keys()))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)

    summary = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "detector": detector_name,
        "images": len(images),
        "crops": total_crops,
        "no_detection_images": no_detection_count,
        "conf": conf,
        "iou": iou,
        "imgsz": imgsz,
        "allowed_detector_classes": sorted(allowed_detector_classes),
    }
    (metadata_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logging.info("Finished YOLO crop generation: %s crops, %s no-detection images", total_crops, no_detection_count)
    logging.info("Crops root: %s", crops_root)


if __name__ == "__main__":
    main()






