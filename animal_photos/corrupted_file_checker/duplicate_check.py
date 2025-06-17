import os
from PIL import Image
import imagehash
import shutil

# Hoofdmap met submappen per dier
BASE_DIR = '/mnt/e/MachineLearning/animal_photos/simple_images'

# Centrale duplicatenmap
CENTRAL_DUP_DIR = '/mnt/e/MachineLearning/animal_photos/duplicates'
os.makedirs(CENTRAL_DUP_DIR, exist_ok=True)

THRESHOLD = 5  # Tolerantie voor hash-vergelijking

def check_duplicates_in_folder(folder_path, category_name):
    print(f"\n🔍 Controleren op duplicaten in: {category_name}")
    hashes = {}

    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)

        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')) and os.path.isfile(filepath):
            try:
                img = Image.open(filepath)
                img_hash = imagehash.phash(img)

                duplicate_found = False
                for other_file, other_hash in hashes.items():
                    if abs(img_hash - other_hash) <= THRESHOLD:
                        # Nieuwe naam maken met categorie als prefix om conflicten te voorkomen
                        new_name = f"{category_name}__{filename}"
                        target_path = os.path.join(CENTRAL_DUP_DIR, new_name)
                        print(f"  {filename} lijkt op {other_file} → verplaatst naar {new_name}")
                        shutil.move(filepath, target_path)
                        duplicate_found = True
                        break

                if not duplicate_found:
                    hashes[filename] = img_hash

            except Exception as e:
                print(f"  ⚠️ Fout bij openen van {filename}: {e}")

if __name__ == "__main__":
    for category in os.listdir(BASE_DIR):
        category_path = os.path.join(BASE_DIR, category)
        if os.path.isdir(category_path):
            check_duplicates_in_folder(category_path, category)

    print("\n✅ Alle duplicaten zijn verplaatst naar:", CENTRAL_DUP_DIR)
