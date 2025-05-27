import os
from PIL import Image
import imagehash
import shutil

# Map met afbeeldingen
DIRECTORY = '/mnt/e/MachineLearning/animal_photos/simple_images/badger'

# Map waar duplicaten heen worden verplaatst
DUPLICATE_DIR = os.path.join(DIRECTORY, 'duplicates')

# Hoeveel verschil toegestaan is tussen hashes
THRESHOLD = 5

# Zorg dat de duplicatenmap bestaat
os.makedirs(DUPLICATE_DIR, exist_ok=True)

# Opslag voor unieke hashes
hashes = {}

# Loop door alle afbeeldingen
for filename in os.listdir(DIRECTORY):
    filepath = os.path.join(DIRECTORY, filename)

    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp')) and os.path.isfile(filepath):
        try:
            img = Image.open(filepath)
            img_hash = imagehash.phash(img)

            # Vergelijk met eerdere hashes
            duplicate_found = False
            for other_file, other_hash in hashes.items():
                if abs(img_hash - other_hash) <= THRESHOLD:
                    print(f"{filename} lijkt op {other_file} → verplaatsen naar duplicates/")
                    shutil.move(filepath, os.path.join(DUPLICATE_DIR, filename))
                    duplicate_found = True
                    break

            if not duplicate_found:
                hashes[filename] = img_hash

        except Exception as e:
            print(f"Fout bij openen van {filename}: {e}")
