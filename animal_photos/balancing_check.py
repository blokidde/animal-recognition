import argparse
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def count_images(base_dir: Path) -> dict[str, int]:
    if not base_dir.exists():
        raise FileNotFoundError(f"Datasetmap bestaat niet: {base_dir}")

    class_counts = {}
    for class_dir in sorted(p for p in base_dir.iterdir() if p.is_dir()):
        class_counts[class_dir.name] = sum(
            1
            for image_path in class_dir.iterdir()
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS
        )
    return class_counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Controleer hoeveel afbeeldingen er per klasse zijn.")
    parser.add_argument(
        "--base_dir",
        default="animal_photos/simple_images",
        help="Datasetmap met een submap per klasse.",
    )
    args = parser.parse_args()

    print(count_images(Path(args.base_dir)))


if __name__ == "__main__":
    main()
