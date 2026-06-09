import argparse
import json
import tensorflow as tf
import numpy as np
from pathlib import Path
from tensorflow.keras.preprocessing import image

img_size = (224, 224)

parser = argparse.ArgumentParser(description="Controleer kans op een opgegeven klasse voor een map afbeeldingen.")
parser.add_argument("--model_path", default="tested_models/working_model_1_10_25/custom_mobilenetv3large_model.keras")
parser.add_argument("--images_dir", required=True, help="Map met testfoto's.")
parser.add_argument("--class_indices", default="train/class_indices.json")
parser.add_argument("--class_name", default="badger")
args = parser.parse_args()

# === Model laden ===
model = tf.keras.models.load_model(args.model_path)

with open(args.class_indices, "r", encoding="utf-8") as f:
    class_indices = json.load(f)
class_names = [name for name, _ in sorted(class_indices.items(), key=lambda item: item[1])]
print("Klassenvolgorde:", class_names)

if args.class_name not in class_names:
    raise ValueError(f"'{args.class_name}' staat niet in je trainingsklassen. Klassen gevonden: {class_names}")
target_idx = class_names.index(args.class_name)

# === Functie om kans op das te berekenen ===
def predict_class(img_path):
    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array, verbose=0)
    return preds[0][target_idx]

# === Loop door alle foto's in de map ===
for image_path in Path(args.images_dir).iterdir():
    if image_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
        prob = predict_class(image_path)
        print(f"{image_path.name} → kans op {args.class_name}: {prob:.2f}")
