import argparse
from pathlib import Path
from PIL import Image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def check_and_remove_corrupt_images(directory: Path, remove: bool) -> None:
    if not directory.exists():
        raise FileNotFoundError(f"Datasetmap bestaat niet: {directory}")

    for file_path in directory.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
            try:
                # Probeer het bestand als afbeelding te openen
                with Image.open(file_path) as img:
                    img.verify()  # Controleer op corruptie
            except (IOError, SyntaxError):
                if remove:
                    print(f"Corrupte afbeelding gevonden en verwijderd: {file_path}")
                    file_path.unlink()
                else:
                    print(f"Corrupte afbeelding gevonden: {file_path}")

def main() -> None:
    parser = argparse.ArgumentParser(description="Controleer datasetafbeeldingen op corruptie.")
    parser.add_argument(
        "--dataset_path",
        default="animal_photos/simple_images",
        help="Datasetmap met afbeeldingen.",
    )
    parser.add_argument(
        "--remove",
        action="store_true",
        help="Verwijder corrupte afbeeldingen. Zonder deze vlag wordt alleen gemeld.",
    )
    args = parser.parse_args()
    check_and_remove_corrupt_images(Path(args.dataset_path), args.remove)


if __name__ == "__main__":
    main()
