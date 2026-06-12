from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import random
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
DEFAULT_CONFIG = CURRENT_DIR / "params_yolo26_detector.yaml"

DUTCH_MONTH_NAMES = {
    1: "januari",
    2: "februari",
    3: "maart",
    4: "april",
    5: "mei",
    6: "juni",
    7: "juli",
    8: "augustus",
    9: "september",
    10: "oktober",
    11: "november",
    12: "december",
}


@dataclass(frozen=True)
class SplitCounts:
    species: str
    class_id: int
    train: int
    train_original: int
    train_oversampled: int
    val: int
    test: int
    total: int


@dataclass(frozen=True)
class DetectionSample:
    source_path: Path
    species: str
    x_center: float
    y_center: float
    width: float
    height: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a normal YOLO detection model on DeepFaune-cropped animal images.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--prepare-only", action="store_true", help="Only build/update the YOLO detection dataset.")
    parser.add_argument("--resume", action="store_true", help="Resume an interrupted Ultralytics run.")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config does not exist: {path}")
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return data


def resolve_path(value: str | Path, base: Path = PROJECT_ROOT) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def normalize_model_name(model_name: str) -> str:
    normalized = model_name.strip()
    if normalized == "":
        return "yolo26n.pt"

    lowered = normalized.lower()
    shorthand_map = {
        "yolo11n": "yolo11n.pt",
        "yolo11s": "yolo11s.pt",
        "yolo26n": "yolo26n.pt",
        "yolo26s": "yolo26s.pt",
        "yolo26m": "yolo26m.pt",
        "yolo26l": "yolo26l.pt",
        "yolo26x": "yolo26x.pt",
    }
    if lowered in shorthand_map:
        return shorthand_map[lowered]
    if lowered.endswith("-seg.pt") or lowered.endswith("-cls.pt"):
        raise ValueError("Use a normal YOLO detection checkpoint, for example yolo26n.pt, not -seg or -cls.")
    return normalized


def model_output_folder_name(model_name: str) -> str:
    return Path(model_name).stem


def format_dutch_date_label(day_value: date | None = None) -> str:
    current_day = day_value or date.today()
    return f"{current_day.day}-{current_day.month}-{str(current_day.year)[-2:]}"


def get_next_run_name(project_dir: Path, date_label: str) -> str:
    project_dir.mkdir(parents=True, exist_ok=True)
    new_pattern = re.compile(rf"^run {re.escape(date_label)} (\d+)$")
    old_pattern = re.compile(rf"^run (\d+) {re.escape(date_label)}$")
    max_run = 0
    for path in project_dir.iterdir():
        if not path.is_dir():
            continue
        match = new_pattern.match(path.name) or old_pattern.match(path.name)
        if match:
            max_run = max(max_run, int(match.group(1)))
    return f"run {date_label} {max_run + 1}"


def list_species_images(source_root: Path) -> dict[str, list[Path]]:
    species_images: dict[str, list[Path]] = {}
    for species_dir in sorted(path for path in source_root.iterdir() if path.is_dir()):
        images = sorted(
            path for path in species_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if images:
            species_images[species_dir.name] = images
    return species_images


def split_images(images: list[Path], train_ratio: float, val_ratio: float) -> tuple[list[Path], list[Path], list[Path]]:
    total = len(images)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    return images[:train_end], images[train_end:val_end], images[val_end:]


def safe_stem(value: str, max_length: int = 90) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned[:max_length] or "image"


def output_image_name(species: str, source_path: Path) -> str:
    digest = hashlib.sha1(str(source_path).encode("utf-8")).hexdigest()[:10]
    return f"{safe_stem(species)}__{safe_stem(source_path.stem)}__{digest}{source_path.suffix.lower()}"


def place_file(source: Path, destination: Path, link_method: str) -> None:
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if link_method == "hardlink":
        try:
            os.link(source, destination)
            return
        except OSError as exc:
            logging.warning("Hardlink failed for %s; falling back to copy: %s", source, exc)
    shutil.copy2(source, destination)


def write_detection_sample(
    source_path: Path,
    image_path: Path,
    label_path: Path,
    class_id: int,
    link_method: str,
    bbox: tuple[float, float, float, float] = (0.5, 0.5, 1.0, 1.0),
) -> None:
    place_file(source_path, image_path, link_method)
    x_center, y_center, width, height = bbox
    label_path.write_text(
        f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n",
        encoding="utf-8",
    )


def get_oversample_target(config: dict[str, Any], train_counts: dict[str, int]) -> int:
    configured = optional_value(config, "oversample_target_train_images")
    if configured is not None:
        return int(configured)
    return max(train_counts.values()) if train_counts else 0


def get_display_class_names(class_names: list[str], config: dict[str, Any]) -> list[str]:
    class_name_map = config.get("class_name_map", {}) or {}
    if not isinstance(class_name_map, dict):
        raise ValueError("class_name_map must be a YAML mapping from scientific names to display names")
    return [str(class_name_map.get(name, name)) for name in class_names]


def write_data_yaml(prepared_root: Path, class_names: list[str]) -> Path:
    data_yaml_path = prepared_root / "data.yaml"
    data = {
        "path": str(prepared_root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {class_id: name for class_id, name in enumerate(class_names)},
    }
    with data_yaml_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, sort_keys=False)
    return data_yaml_path


def prepare_detection_dataset(config: dict[str, Any]) -> Path:
    source_root = resolve_path(config["source_root"])
    prepared_root = resolve_path(config["prepared_root"])
    train_ratio = float(config.get("train_ratio", 0.70))
    val_ratio = float(config.get("val_ratio", 0.15))
    test_ratio = float(config.get("test_ratio", 0.15))
    seed = int(config.get("seed", 42))
    link_method = str(config.get("link_method", "hardlink"))
    regenerate = bool(config.get("regenerate_dataset", False))
    oversample_train_classes = bool(config.get("oversample_train_classes", True))
    oversample_max_extra_copies = int(config.get("oversample_max_extra_copies", 5))

    if not source_root.exists():
        raise FileNotFoundError(f"Source dataset does not exist: {source_root}")
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 0.001:
        raise ValueError("train_ratio + val_ratio + test_ratio must be 1.0")
    if link_method not in {"hardlink", "copy"}:
        raise ValueError("link_method must be hardlink or copy")

    if regenerate and prepared_root.exists():
        shutil.rmtree(prepared_root)

    metadata_dir = prepared_root / "_metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    species_images = list_species_images(source_root)
    if not species_images:
        raise RuntimeError(f"No class folders with images found in {source_root}")

    class_names = sorted(species_images)
    display_class_names = get_display_class_names(class_names, config)
    class_ids = {species: class_id for class_id, species in enumerate(class_names)}
    split_plan: dict[str, tuple[list[Path], list[Path], list[Path]]] = {}
    rng = random.Random(seed)

    for species in class_names:
        images = species_images[species][:]
        rng.shuffle(images)
        split_plan[species] = split_images(images, train_ratio, val_ratio)

    train_counts = {species: len(splits[0]) for species, splits in split_plan.items()}
    oversample_target = get_oversample_target(config, train_counts) if oversample_train_classes else 0
    logging.info(
        "Train oversampling: enabled=%s target=%s max_extra_copies=%s",
        oversample_train_classes,
        oversample_target,
        oversample_max_extra_copies,
    )

    counts: list[SplitCounts] = []
    manifest_rows: list[dict[str, str | int]] = []

    for species in class_names:
        train_images, val_images, test_images = split_plan[species]
        split_map = {"train": train_images, "val": val_images, "test": test_images}
        class_id = class_ids[species]
        oversampled_count = 0

        for split_name, split_files in split_map.items():
            image_dir = prepared_root / "images" / split_name
            label_dir = prepared_root / "labels" / split_name
            image_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)

            for source_path in split_files:
                image_name = output_image_name(species, source_path)
                image_path = image_dir / image_name
                label_path = label_dir / f"{Path(image_name).stem}.txt"
                write_detection_sample(source_path, image_path, label_path, class_id, link_method)
                manifest_rows.append(
                    {
                        "split": split_name,
                        "species": species,
                        "class_id": class_id,
                        "source_path": str(source_path),
                        "image_path": str(image_path),
                        "label_path": str(label_path),
                    }
                )

        if oversample_train_classes and train_images and len(train_images) < oversample_target:
            max_extra = len(train_images) * max(0, oversample_max_extra_copies)
            needed = min(oversample_target - len(train_images), max_extra)
            train_image_dir = prepared_root / "images" / "train"
            train_label_dir = prepared_root / "labels" / "train"
            for oversample_index in range(needed):
                source_path = rng.choice(train_images)
                original_name = output_image_name(species, source_path)
                image_stem = Path(original_name).stem
                image_name = f"{image_stem}__os{oversample_index + 1:04d}{source_path.suffix.lower()}"
                image_path = train_image_dir / image_name
                label_path = train_label_dir / f"{Path(image_name).stem}.txt"
                write_detection_sample(source_path, image_path, label_path, class_id, link_method)
                manifest_rows.append(
                    {
                        "split": "train",
                        "species": species,
                        "class_id": class_id,
                        "source_path": str(source_path),
                        "image_path": str(image_path),
                        "label_path": str(label_path),
                        "oversampled": 1,
                    }
                )
            oversampled_count = needed

        counts.append(
            SplitCounts(
                species=species,
                class_id=class_id,
                train=len(train_images) + oversampled_count,
                train_original=len(train_images),
                train_oversampled=oversampled_count,
                val=len(val_images),
                test=len(test_images),
                total=len(species_images[species]),
            )
        )
        logging.info(
            "%s: class_id=%s train=%s original_train=%s oversampled=%s val=%s test=%s total=%s",
            species,
            class_id,
            len(train_images) + oversampled_count,
            len(train_images),
            oversampled_count,
            len(val_images),
            len(test_images),
            len(species_images[species]),
        )

    (metadata_dir / "split_counts.json").write_text(
        json.dumps([asdict(item) for item in counts], indent=2),
        encoding="utf-8",
    )
    (metadata_dir / "class_names.json").write_text(
        json.dumps(class_ids, indent=2),
        encoding="utf-8",
    )
    (metadata_dir / "display_class_names.json").write_text(
        json.dumps({name: display for name, display in zip(class_names, display_class_names)}, indent=2),
        encoding="utf-8",
    )
    (metadata_dir / "split_manifest.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest_rows),
        encoding="utf-8",
    )
    data_yaml_path = write_data_yaml(prepared_root, display_class_names)
    logging.info("Prepared %s classes with %s images", len(counts), sum(item.total for item in counts))
    logging.info("YOLO data yaml: %s", data_yaml_path)
    return data_yaml_path



def clamp01(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def bbox_from_deepfaune_row(row: dict[str, str]) -> tuple[float, float, float, float] | None:
    image_width = float(row["image_width"])
    image_height = float(row["image_height"])
    x1 = clamp01(float(row["x1"]) / image_width)
    y1 = clamp01(float(row["y1"]) / image_height)
    x2 = clamp01(float(row["x2"]) / image_width)
    y2 = clamp01(float(row["y2"]) / image_height)
    width = x2 - x1
    height = y2 - y1
    if width <= 0.0 or height <= 0.0:
        return None
    x_center = x1 + width / 2.0
    y_center = y1 + height / 2.0
    return clamp01(x_center), clamp01(y_center), clamp01(width), clamp01(height)


def load_deepfaune_samples(metadata_csv: Path) -> dict[str, list[DetectionSample]]:
    if not metadata_csv.exists():
        raise FileNotFoundError(f"DeepFaune metadata CSV does not exist: {metadata_csv}")

    samples: dict[str, list[DetectionSample]] = {}
    skipped_missing = 0
    skipped_invalid_bbox = 0

    with metadata_csv.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        required = {"normal_output_path", "species", "x1", "y1", "x2", "y2", "image_width", "image_height"}
        missing_columns = required - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"Missing required metadata columns in {metadata_csv}: {sorted(missing_columns)}")

        for row in reader:
            source_path = Path(row["normal_output_path"])
            if not source_path.exists():
                skipped_missing += 1
                continue
            bbox = bbox_from_deepfaune_row(row)
            if bbox is None:
                skipped_invalid_bbox += 1
                continue
            species = row["species"]
            samples.setdefault(species, []).append(
                DetectionSample(
                    source_path=source_path,
                    species=species,
                    x_center=bbox[0],
                    y_center=bbox[1],
                    width=bbox[2],
                    height=bbox[3],
                )
            )

    if skipped_missing:
        logging.warning("Skipped %s metadata rows with missing normal images", skipped_missing)
    if skipped_invalid_bbox:
        logging.warning("Skipped %s metadata rows with invalid DeepFaune bboxes", skipped_invalid_bbox)
    return samples


def output_sample_name(species: str, source_path: Path) -> str:
    digest = hashlib.sha1(str(source_path).encode("utf-8")).hexdigest()[:10]
    return f"{safe_stem(species)}__{safe_stem(source_path.stem)}__{digest}{source_path.suffix.lower()}"


def prepare_deepfaune_box_dataset(config: dict[str, Any]) -> Path:
    metadata_csv = resolve_path(config["metadata_csv"])
    prepared_root = resolve_path(config["prepared_root"])
    train_ratio = float(config.get("train_ratio", 0.70))
    val_ratio = float(config.get("val_ratio", 0.15))
    test_ratio = float(config.get("test_ratio", 0.15))
    seed = int(config.get("seed", 42))
    link_method = str(config.get("link_method", "hardlink"))
    regenerate = bool(config.get("regenerate_dataset", False))
    oversample_train_classes = bool(config.get("oversample_train_classes", True))
    oversample_max_extra_copies = int(config.get("oversample_max_extra_copies", 5))

    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 0.001:
        raise ValueError("train_ratio + val_ratio + test_ratio must be 1.0")
    if link_method not in {"hardlink", "copy"}:
        raise ValueError("link_method must be hardlink or copy")
    if regenerate and prepared_root.exists():
        shutil.rmtree(prepared_root)

    metadata_dir = prepared_root / "_metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    species_samples = load_deepfaune_samples(metadata_csv)
    if not species_samples:
        raise RuntimeError(f"No DeepFaune samples found in {metadata_csv}")

    class_names = sorted(species_samples)
    display_class_names = get_display_class_names(class_names, config)
    class_ids = {species: class_id for class_id, species in enumerate(class_names)}
    rng = random.Random(seed)
    split_plan: dict[str, tuple[list[DetectionSample], list[DetectionSample], list[DetectionSample]]] = {}

    for species in class_names:
        samples = species_samples[species][:]
        rng.shuffle(samples)
        train_samples, val_samples, test_samples = split_images(samples, train_ratio, val_ratio)
        split_plan[species] = train_samples, val_samples, test_samples

    train_counts = {species: len(splits[0]) for species, splits in split_plan.items()}
    oversample_target = get_oversample_target(config, train_counts) if oversample_train_classes else 0
    logging.info(
        "Train oversampling: enabled=%s target=%s max_extra_copies=%s",
        oversample_train_classes,
        oversample_target,
        oversample_max_extra_copies,
    )

    counts: list[SplitCounts] = []
    manifest_rows: list[dict[str, str | int | float]] = []

    for species in class_names:
        train_samples, val_samples, test_samples = split_plan[species]
        split_map = {"train": train_samples, "val": val_samples, "test": test_samples}
        class_id = class_ids[species]
        oversampled_count = 0

        for split_name, split_samples in split_map.items():
            image_dir = prepared_root / "images" / split_name
            label_dir = prepared_root / "labels" / split_name
            image_dir.mkdir(parents=True, exist_ok=True)
            label_dir.mkdir(parents=True, exist_ok=True)

            for sample in split_samples:
                image_name = output_sample_name(species, sample.source_path)
                image_path = image_dir / image_name
                label_path = label_dir / f"{Path(image_name).stem}.txt"
                bbox = (sample.x_center, sample.y_center, sample.width, sample.height)
                write_detection_sample(sample.source_path, image_path, label_path, class_id, link_method, bbox)
                manifest_rows.append(
                    {
                        "split": split_name,
                        "species": species,
                        "class_id": class_id,
                        "source_path": str(sample.source_path),
                        "image_path": str(image_path),
                        "label_path": str(label_path),
                        "x_center": sample.x_center,
                        "y_center": sample.y_center,
                        "width": sample.width,
                        "height": sample.height,
                    }
                )

        if oversample_train_classes and train_samples and len(train_samples) < oversample_target:
            max_extra = len(train_samples) * max(0, oversample_max_extra_copies)
            needed = min(oversample_target - len(train_samples), max_extra)
            train_image_dir = prepared_root / "images" / "train"
            train_label_dir = prepared_root / "labels" / "train"
            for oversample_index in range(needed):
                sample = rng.choice(train_samples)
                original_name = output_sample_name(species, sample.source_path)
                image_stem = Path(original_name).stem
                image_name = f"{image_stem}__os{oversample_index + 1:04d}{sample.source_path.suffix.lower()}"
                image_path = train_image_dir / image_name
                label_path = train_label_dir / f"{Path(image_name).stem}.txt"
                bbox = (sample.x_center, sample.y_center, sample.width, sample.height)
                write_detection_sample(sample.source_path, image_path, label_path, class_id, link_method, bbox)
                manifest_rows.append(
                    {
                        "split": "train",
                        "species": species,
                        "class_id": class_id,
                        "source_path": str(sample.source_path),
                        "image_path": str(image_path),
                        "label_path": str(label_path),
                        "x_center": sample.x_center,
                        "y_center": sample.y_center,
                        "width": sample.width,
                        "height": sample.height,
                        "oversampled": 1,
                    }
                )
            oversampled_count = needed

        counts.append(
            SplitCounts(
                species=species,
                class_id=class_id,
                train=len(train_samples) + oversampled_count,
                train_original=len(train_samples),
                train_oversampled=oversampled_count,
                val=len(val_samples),
                test=len(test_samples),
                total=len(species_samples[species]),
            )
        )
        logging.info(
            "%s: class_id=%s train=%s original_train=%s oversampled=%s val=%s test=%s total=%s",
            species,
            class_id,
            len(train_samples) + oversampled_count,
            len(train_samples),
            oversampled_count,
            len(val_samples),
            len(test_samples),
            len(species_samples[species]),
        )

    (metadata_dir / "split_counts.json").write_text(
        json.dumps([asdict(item) for item in counts], indent=2),
        encoding="utf-8",
    )
    (metadata_dir / "class_names.json").write_text(json.dumps(class_ids, indent=2), encoding="utf-8")
    (metadata_dir / "display_class_names.json").write_text(
        json.dumps({name: display for name, display in zip(class_names, display_class_names)}, indent=2),
        encoding="utf-8",
    )
    (metadata_dir / "split_manifest.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest_rows),
        encoding="utf-8",
    )
    data_yaml_path = write_data_yaml(prepared_root, display_class_names)
    logging.info("Prepared %s classes with %s images", len(counts), sum(item.total for item in counts))
    logging.info("YOLO data yaml: %s", data_yaml_path)
    return data_yaml_path


def prepare_dataset(config: dict[str, Any]) -> Path:
    dataset_mode = str(config.get("dataset_mode", "fullbox_crops")).strip().lower()
    if dataset_mode == "fullbox_crops":
        return prepare_detection_dataset(config)
    if dataset_mode == "deepfaune_boxes":
        return prepare_deepfaune_box_dataset(config)
    raise ValueError("dataset_mode must be 'fullbox_crops' or 'deepfaune_boxes'")

def optional_value(config: dict[str, Any], key: str) -> Any | None:
    value = config.get(key)
    if value is None or str(value).strip() == "":
        return None
    return value


def train_yolo(config: dict[str, Any], data_yaml_path: Path, resume: bool) -> None:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Ultralytics is missing. Run: uv sync") from exc

    model_name = normalize_model_name(str(config.get("model_name", "yolo26n.pt")))
    model_path = Path(model_name)
    model_source = str(model_path.resolve()) if model_path.exists() else model_name

    checkpoints_root = resolve_path(config.get("checkpoints_root", "checkpoints/detection_model"))
    project_dir = checkpoints_root / model_output_folder_name(model_name)
    run_name = str(config.get("run_name", "")).strip() or get_next_run_name(project_dir, format_dutch_date_label())

    logging.info("Starting YOLO detection training with model: %s", model_source)
    logging.info("Using dataset yaml: %s", data_yaml_path)
    logging.info("Saving run to: %s", project_dir / run_name)

    train_kwargs: dict[str, Any] = {
        "data": str(data_yaml_path),
        "epochs": int(config.get("epochs", 100)),
        "imgsz": int(config.get("image_size", 512)),
        "batch": int(config.get("batch_size", 16)),
        "workers": int(config.get("num_workers", 4)),
        "amp": bool(config.get("amp", True)),
        "cache": bool(config.get("cache", False)),
        "patience": int(config.get("patience", 25)),
        "project": str(project_dir),
        "name": run_name,
        "verbose": True,
    }

    device = optional_value(config, "device")
    if device is not None:
        train_kwargs["device"] = str(device)

    for key in (
        "lr0",
        "lrf",
        "weight_decay",
        "warmup_epochs",
        "degrees",
        "translate",
        "scale",
        "fliplr",
        "hsv_h",
        "hsv_s",
        "hsv_v",
        "mosaic",
        "close_mosaic",
        "mixup",
        "copy_paste",
    ):
        value = optional_value(config, key)
        if value is not None:
            train_kwargs[key] = value

    optimizer = optional_value(config, "optimizer")
    if optimizer is not None:
        train_kwargs["optimizer"] = str(optimizer)

    cos_lr = config.get("cos_lr")
    if cos_lr is not None:
        train_kwargs["cos_lr"] = bool(cos_lr)

    if bool(config.get("resume", False)) or resume:
        train_kwargs["resume"] = True

    model = YOLO(model_source)
    model.train(**train_kwargs)
    logging.info("YOLO detection training finished")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(args.config)
    data_yaml_path = prepare_dataset(config)
    if args.prepare_only:
        logging.info("Stopping because --prepare-only was set")
        return
    train_yolo(config, data_yaml_path, args.resume)


if __name__ == "__main__":
    main()