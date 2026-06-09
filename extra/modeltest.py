import argparse
import json
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

img_size = (224, 224)

parser = argparse.ArgumentParser(description="Toon alle klassekansen voor een enkele afbeelding.")
parser.add_argument("--model_path", default="tested_models/working_model_31_08_25/custom_mobilenetv3large_model.keras")
parser.add_argument("--img_path", required=True)
parser.add_argument("--class_indices", default="train/class_indices.json")
args = parser.parse_args()

# Laden van model
model = tf.keras.models.load_model(args.model_path)

with open(args.class_indices, "r", encoding="utf-8") as f:
    class_indices = json.load(f)
class_names = [name for name, _ in sorted(class_indices.items(), key=lambda item: item[1])]

img = image.load_img(args.img_path, target_size=img_size)
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

preds = model.predict(img_array, verbose=0)[0]

print(f"Voorspellingen voor {os.path.basename(args.img_path)}:\n")
for cname, prob in zip(class_names, preds):
    print(f"{cname:15s}: {prob:.4f}")
