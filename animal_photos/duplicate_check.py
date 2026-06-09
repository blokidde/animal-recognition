import os
import argparse
from PIL import Image
import imagehash
import shutil
from pathlib import Path

THRESHOLD = 1  # Tolerantie voor hash-vergelijking
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

def check_duplicates_in_folder(folder_path, category_name, duplicate_dir):
    print(f"\n🔍 Controleren op duplicaten in: {category_name}")
    hashes = {}

    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)

        if filename.lower().endswith(IMAGE_EXTENSIONS) and os.path.isfile(filepath):
            try:
                img = Image.open(filepath)
                img_hash = imagehash.phash(img)

                duplicate_found = False
                for other_file, other_hash in hashes.items():
                    if abs(img_hash - other_hash) <= THRESHOLD:
                        # Nieuwe naam maken met categorie als prefix om conflicten te voorkomen
                        new_name = f"{category_name}__{filename}"
                        target_path = os.path.join(duplicate_dir, new_name)
                        print(f"  {filename} lijkt op {other_file} → verplaatst naar {new_name}")
                        shutil.move(filepath, target_path)
                        duplicate_found = True
                        break

                if not duplicate_found:
                    hashes[filename] = img_hash

            except Exception as e:
                print(f"  ⚠️ Fout bij openen van {filename}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verplaats dubbele afbeeldingen naar een centrale map.")
    parser.add_argument(
        "--base_dir",
        default="animal_photos/simple_images",
        help="Datasetmap met een submap per klasse.",
    )
    parser.add_argument(
        "--duplicates_dir",
        default="animal_photos/duplicates",
        help="Map waar gevonden duplicaten naartoe worden verplaatst.",
    )
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    duplicate_dir = Path(args.duplicates_dir)
    duplicate_dir.mkdir(parents=True, exist_ok=True)

    if not base_dir.exists():
        raise FileNotFoundError(f"Datasetmap bestaat niet: {base_dir}")

    for category in os.listdir(base_dir):
        category_path = os.path.join(base_dir, category)
        if os.path.isdir(category_path):
            check_duplicates_in_folder(category_path, category, str(duplicate_dir))

    print("\n✅ Alle duplicaten zijn verplaatst naar:", duplicate_dir)
