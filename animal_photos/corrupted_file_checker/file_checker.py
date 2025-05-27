import os
from PIL import Image

def check_and_remove_corrupt_images(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                # Probeer het bestand als afbeelding te openen
                with Image.open(file_path) as img:
                    img.verify()  # Controleer op corruptie
            except (IOError, SyntaxError):
                print(f"Corrupte afbeelding gevonden en verwijderd: {file_path}")
                os.remove(file_path)

# Pad naar je dataset
dataset_path = '/home/jurriaan/animalrec/animal_photos/simple_images'
check_and_remove_corrupt_images(dataset_path)
