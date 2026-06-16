from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any
try:
    from windows_platform_patch import patch_windows_platform_wmi
except ImportError:
    from classifier.windows_platform_patch import patch_windows_platform_wmi

patch_windows_platform_wmi()

import torch
import torch.nn as nn
import yaml
from PIL import Image, ImageOps
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
DEFAULT_CONFIG = CURRENT_DIR / "params_resnet18_classifier.yaml"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class SplitCounts:
    species: str
    class_id: int
    train: int
    val: int
    test: int
    total: int


@dataclass(frozen=True)
class ImageSample:
    path: Path
    species: str
    class_id: int


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
        left = (self.size - new_width) // 2
        top = (self.size - new_height) // 2
        canvas.paste(image, (left, top))
        return canvas


class ClassificationDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(self, samples: list[ImageSample], transform: transforms.Compose) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[index]
        with Image.open(sample.path) as image:
            tensor = self.transform(image)
        return tensor, sample.class_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a ResNet18 animal species classifier on DeepFaune crops.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", type=Path, default=None, help="Path to last.pt to resume training.")
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
    return path if path.is_absolute() else (base / path).resolve()


def format_date_label(day_value: date | None = None) -> str:
    current_day = day_value or date.today()
    return f"{current_day.day}-{current_day.month}-{str(current_day.year)[-2:]}"


def get_next_run_name(project_dir: Path, date_label: str) -> str:
    project_dir.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(rf"^run {re.escape(date_label)} (\d+)$")
    max_run = 0
    for path in project_dir.iterdir():
        if path.is_dir():
            match = pattern.match(path.name)
            if match:
                max_run = max(max_run, int(match.group(1)))
    return f"run {date_label} {max_run + 1}"


def safe_stem(value: str, max_length: int = 90) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned[:max_length] or "image"


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


def split_paths(paths: list[Path], train_ratio: float, val_ratio: float) -> tuple[list[Path], list[Path], list[Path]]:
    train_end = int(len(paths) * train_ratio)
    val_end = train_end + int(len(paths) * val_ratio)
    return paths[:train_end], paths[train_end:val_end], paths[val_end:]


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


def get_display_class_names(class_names: list[str], config: dict[str, Any]) -> list[str]:
    class_name_map = config.get("class_name_map", {}) or {}
    if not isinstance(class_name_map, dict):
        raise ValueError("class_name_map must be a YAML mapping")
    return [str(class_name_map.get(name, name)) for name in class_names]


def prepare_dataset(config: dict[str, Any]) -> Path:
    source_root = resolve_path(config["source_root"])
    prepared_root = resolve_path(config["prepared_root"])
    train_ratio = float(config.get("train_ratio", 0.70))
    val_ratio = float(config.get("val_ratio", 0.15))
    test_ratio = float(config.get("test_ratio", 0.15))
    seed = int(config.get("seed", 42))
    link_method = str(config.get("link_method", "hardlink"))
    regenerate = bool(config.get("regenerate_dataset", False))

    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.001:
        raise ValueError("train_ratio + val_ratio + test_ratio must be 1.0")
    if link_method not in {"hardlink", "copy"}:
        raise ValueError("link_method must be hardlink or copy")
    if not source_root.exists():
        raise FileNotFoundError(f"Source crop dataset does not exist: {source_root}")
    if regenerate and prepared_root.exists():
        shutil.rmtree(prepared_root)

    metadata_dir = prepared_root / "_metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    species_images = list_species_images(source_root)
    if not species_images:
        raise RuntimeError(f"No class folders with images found in {source_root}")

    class_names = sorted(species_images)
    display_names = get_display_class_names(class_names, config)
    class_ids = {species: idx for idx, species in enumerate(class_names)}
    rng = random.Random(seed)
    counts: list[SplitCounts] = []
    manifest_rows: list[dict[str, str | int]] = []

    for species in class_names:
        paths = species_images[species][:]
        rng.shuffle(paths)
        train_paths, val_paths, test_paths = split_paths(paths, train_ratio, val_ratio)
        split_map = {"train": train_paths, "val": val_paths, "test": test_paths}
        class_id = class_ids[species]

        for split_name, split_paths_value in split_map.items():
            target_dir = prepared_root / split_name / species
            target_dir.mkdir(parents=True, exist_ok=True)
            for source_path in split_paths_value:
                destination = target_dir / f"{safe_stem(source_path.stem)}{source_path.suffix.lower()}"
                place_file(source_path, destination, link_method)
                manifest_rows.append(
                    {
                        "split": split_name,
                        "species": species,
                        "display_name": display_names[class_id],
                        "class_id": class_id,
                        "source_path": str(source_path),
                        "image_path": str(destination),
                    }
                )

        counts.append(
            SplitCounts(
                species=species,
                class_id=class_id,
                train=len(train_paths),
                val=len(val_paths),
                test=len(test_paths),
                total=len(paths),
            )
        )
        logging.info(
            "%s: class_id=%s train=%s val=%s test=%s total=%s",
            species,
            class_id,
            len(train_paths),
            len(val_paths),
            len(test_paths),
            len(paths),
        )

    (metadata_dir / "class_names.json").write_text(json.dumps(class_ids, indent=2), encoding="utf-8")
    (metadata_dir / "display_class_names.json").write_text(
        json.dumps({name: display for name, display in zip(class_names, display_names)}, indent=2),
        encoding="utf-8",
    )
    (metadata_dir / "split_counts.json").write_text(
        json.dumps([asdict(item) for item in counts], indent=2),
        encoding="utf-8",
    )
    (metadata_dir / "split_manifest.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest_rows),
        encoding="utf-8",
    )
    logging.info("Prepared classifier dataset at %s", prepared_root)
    return prepared_root


def load_samples(prepared_root: Path, split: str, class_ids: dict[str, int]) -> list[ImageSample]:
    samples: list[ImageSample] = []
    split_root = prepared_root / split
    for species, class_id in class_ids.items():
        class_dir = split_root / species
        if not class_dir.exists():
            continue
        for path in sorted(class_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append(ImageSample(path=path, species=species, class_id=class_id))
    return samples


def build_transforms(config: dict[str, Any], train: bool) -> transforms.Compose:
    image_size = int(config.get("image_size", 384))
    if train:
        return transforms.Compose(
            [
                Letterbox(image_size),
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(
                        float(config.get("random_resized_crop_scale_min", 0.72)),
                        float(config.get("random_resized_crop_scale_max", 1.0)),
                    ),
                    ratio=(0.85, 1.18),
                ),
                transforms.RandomHorizontalFlip(float(config.get("horizontal_flip", 0.5))),
                transforms.ColorJitter(
                    brightness=float(config.get("color_jitter_brightness", 0.25)),
                    contrast=float(config.get("color_jitter_contrast", 0.25)),
                    saturation=float(config.get("color_jitter_saturation", 0.18)),
                    hue=float(config.get("color_jitter_hue", 0.03)),
                ),
                transforms.RandomGrayscale(p=float(config.get("gray_prob", 0.04))),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=float(config.get("gaussian_blur_prob", 0.08))),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                transforms.RandomErasing(p=float(config.get("random_erasing_prob", 0.08)), scale=(0.02, 0.12), ratio=(0.3, 3.3)),
            ]
        )
    return transforms.Compose(
        [
            Letterbox(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def create_model(num_classes: int, pretrained: bool) -> nn.Module:
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def compute_class_counts(samples: list[ImageSample], num_classes: int) -> list[int]:
    counts = [0 for _ in range(num_classes)]
    for sample in samples:
        counts[sample.class_id] += 1
    return counts


def build_sampler(samples: list[ImageSample], num_classes: int) -> WeightedRandomSampler | None:
    counts = compute_class_counts(samples, num_classes)
    weights = [1.0 / max(1, counts[sample.class_id]) for sample in samples]
    return WeightedRandomSampler(weights=torch.DoubleTensor(weights), num_samples=len(samples), replacement=True)


def build_loss_weights(samples: list[ImageSample], num_classes: int, device: torch.device) -> torch.Tensor:
    counts = compute_class_counts(samples, num_classes)
    total = sum(counts)
    weights = [total / max(1, num_classes * count) for count in counts]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def accuracy_topk(logits: torch.Tensor, targets: torch.Tensor, topk: tuple[int, ...] = (1, 5)) -> list[float]:
    max_k = min(max(topk), logits.shape[1])
    _, pred = logits.topk(max_k, dim=1)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    scores: list[float] = []
    for k in topk:
        k = min(k, logits.shape[1])
        scores.append(correct[:k].reshape(-1).float().sum(0).item() / targets.numel())
    return scores


def make_optimizer(model: nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    lr = float(config.get("lr", 0.0003))
    backbone_lr = float(config.get("backbone_lr", lr * 0.25))
    weight_decay = float(config.get("weight_decay", 0.01))
    classifier_params = list(model.fc.parameters())
    classifier_ids = {id(param) for param in classifier_params}
    backbone_params = [param for param in model.parameters() if id(param) not in classifier_ids]
    return torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": classifier_params, "lr": lr},
        ],
        weight_decay=weight_decay,
    )


def make_scheduler(optimizer: torch.optim.Optimizer, config: dict[str, Any]) -> torch.optim.lr_scheduler.LRScheduler:
    epochs = int(config.get("epochs", 80))
    warmup_epochs = int(config.get("warmup_epochs", 3))
    min_lr_factor = float(config.get("min_lr_factor", 0.02))

    def lr_lambda(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_factor + (1.0 - min_lr_factor) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    scaler: GradScaler | None,
    amp: bool,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    total_items = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with autocast(enabled=amp):
                logits = model(images)
                loss = criterion(logits, targets)
            if training:
                assert optimizer is not None and scaler is not None
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        batch_size = targets.size(0)
        top1, top5 = accuracy_topk(logits.detach(), targets, topk=(1, 5))
        total_loss += loss.item() * batch_size
        total_top1 += top1 * batch_size
        total_top5 += top5 * batch_size
        total_items += batch_size

    return {
        "loss": total_loss / max(1, total_items),
        "top1": total_top1 / max(1, total_items),
        "top5": total_top5 / max(1, total_items),
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: GradScaler,
    epoch: int,
    best_val_top1: float,
    config: dict[str, Any],
    class_names: list[str],
    display_names: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "best_val_top1": best_val_top1,
            "config": config,
            "class_names": class_names,
            "display_names": display_names,
            "image_size": int(config.get("image_size", 384)),
            "mean": IMAGENET_MEAN,
            "std": IMAGENET_STD,
            "architecture": str(config.get("model_name", "resnet18")),
        },
        path,
    )


def train_classifier(config: dict[str, Any], prepared_root: Path, resume_path: Path | None = None) -> Path:
    metadata_dir = prepared_root / "_metadata"
    class_ids = json.loads((metadata_dir / "class_names.json").read_text(encoding="utf-8"))
    display_map = json.loads((metadata_dir / "display_class_names.json").read_text(encoding="utf-8"))
    class_names = [name for name, _ in sorted(class_ids.items(), key=lambda item: item[1])]
    display_names = [display_map.get(name, name) for name in class_names]
    num_classes = len(class_names)

    device_value = str(config.get("device", "0"))
    device = torch.device(f"cuda:{device_value}" if device_value != "cpu" and torch.cuda.is_available() else "cpu")
    amp = bool(config.get("amp", True)) and device.type == "cuda"

    train_samples = load_samples(prepared_root, "train", class_ids)
    val_samples = load_samples(prepared_root, "val", class_ids)
    test_samples = load_samples(prepared_root, "test", class_ids)
    if not train_samples or not val_samples:
        raise RuntimeError("Prepared dataset must contain train and val samples")

    train_transform = build_transforms(config, train=True)
    eval_transform = build_transforms(config, train=False)
    sampler = build_sampler(train_samples, num_classes) if bool(config.get("use_weighted_sampler", True)) else None
    train_loader = DataLoader(
        ClassificationDataset(train_samples, train_transform),
        batch_size=int(config.get("batch_size", 48)),
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=int(config.get("num_workers", 6)),
        pin_memory=device.type == "cuda",
        persistent_workers=int(config.get("num_workers", 6)) > 0,
    )
    val_loader = DataLoader(
        ClassificationDataset(val_samples, eval_transform),
        batch_size=int(config.get("batch_size", 48)),
        shuffle=False,
        num_workers=int(config.get("num_workers", 6)),
        pin_memory=device.type == "cuda",
        persistent_workers=int(config.get("num_workers", 6)) > 0,
    )
    test_loader = DataLoader(
        ClassificationDataset(test_samples, eval_transform),
        batch_size=int(config.get("batch_size", 48)),
        shuffle=False,
        num_workers=int(config.get("num_workers", 6)),
        pin_memory=device.type == "cuda",
        persistent_workers=int(config.get("num_workers", 6)) > 0,
    ) if test_samples else None

    checkpoints_root = resolve_path(config.get("checkpoints_root", "checkpoints/classification_model"))
    model_name = str(config.get("model_name", "resnet18"))
    project_dir = checkpoints_root / model_name
    configured_run_name = str(config.get("run_name", "")).strip()
    if resume_path is not None and not configured_run_name:
        resume_path = resume_path.resolve()
        run_dir = resume_path.parent.parent
        run_name = run_dir.name
    else:
        run_name = configured_run_name or get_next_run_name(project_dir, format_date_label())
        run_dir = project_dir / run_name
    weights_dir = run_dir / "weights"
    run_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "args.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (run_dir / "class_names.json").write_text(json.dumps(class_names, indent=2), encoding="utf-8")
    (run_dir / "display_names.json").write_text(json.dumps(display_names, indent=2), encoding="utf-8")

    model = create_model(num_classes, bool(config.get("pretrained", True))).to(device)
    optimizer = make_optimizer(model, config)
    scheduler = make_scheduler(optimizer, config)
    scaler = GradScaler(enabled=amp)
    loss_weights = build_loss_weights(train_samples, num_classes, device) if bool(config.get("use_class_weighted_loss", True)) else None
    criterion = nn.CrossEntropyLoss(weight=loss_weights, label_smoothing=float(config.get("label_smoothing", 0.05)))

    start_epoch = 0
    best_val_top1 = 0.0
    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val_top1 = float(checkpoint.get("best_val_top1", 0.0))
        logging.info("Resuming from %s at epoch %s", resume_path, start_epoch + 1)

    results_path = run_dir / "results.csv"
    if not results_path.exists() or start_epoch == 0:
        with results_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["epoch", "lr", "train_loss", "train_top1", "train_top5", "val_loss", "val_top1", "val_top5"])

    epochs = int(config.get("epochs", 80))
    patience = int(config.get("patience", 20))
    stale_epochs = 0
    logging.info("Training %s classes on %s", num_classes, device)
    logging.info("Saving classifier run to: %s", run_dir)

    for epoch in range(start_epoch, epochs):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer, scaler, amp)
        val_metrics = run_epoch(model, val_loader, criterion, device, None, None, amp)
        scheduler.step()
        lr = optimizer.param_groups[-1]["lr"]
        is_best = val_metrics["top1"] > best_val_top1
        if is_best:
            best_val_top1 = val_metrics["top1"]
            stale_epochs = 0
        else:
            stale_epochs += 1

        save_checkpoint(weights_dir / "last.pt", model, optimizer, scheduler, scaler, epoch, best_val_top1, config, class_names, display_names)
        if is_best:
            save_checkpoint(weights_dir / "best.pt", model, optimizer, scheduler, scaler, epoch, best_val_top1, config, class_names, display_names)

        with results_path.open("a", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                epoch + 1,
                f"{lr:.8f}",
                f"{train_metrics['loss']:.6f}",
                f"{train_metrics['top1']:.6f}",
                f"{train_metrics['top5']:.6f}",
                f"{val_metrics['loss']:.6f}",
                f"{val_metrics['top1']:.6f}",
                f"{val_metrics['top5']:.6f}",
            ])

        logging.info(
            "Epoch %s/%s train_loss=%.4f train_top1=%.3f val_loss=%.4f val_top1=%.3f val_top5=%.3f best=%.3f",
            epoch + 1,
            epochs,
            train_metrics["loss"],
            train_metrics["top1"],
            val_metrics["loss"],
            val_metrics["top1"],
            val_metrics["top5"],
            best_val_top1,
        )
        if stale_epochs >= patience:
            logging.info("Early stopping after %s stale epochs", stale_epochs)
            break

    if test_loader is not None and (weights_dir / "best.pt").exists():
        checkpoint = torch.load(weights_dir / "best.pt", map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        test_metrics = run_epoch(model, test_loader, criterion, device, None, None, amp)
        (run_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
        logging.info("Test metrics: %s", test_metrics)

    logging.info("Classifier training finished: %s", run_dir)
    return run_dir


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(args.config)
    prepared_root = prepare_dataset(config)
    if args.prepare_only:
        logging.info("Stopping because --prepare-only was set")
        return
    train_classifier(config, prepared_root, args.resume)


if __name__ == "__main__":
    main()


