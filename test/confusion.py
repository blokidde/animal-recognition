#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import argparse
import json
import os

# ----------------- Argumenten -----------------
parser = argparse.ArgumentParser(description="Genereer een confusion matrix van een getraind model")
parser.add_argument("--data_dir", "-d", type=str, required=True,
                    help="Root directory met dataset (zelfde structuur als training)")
parser.add_argument("--model_path", "-m", type=str, required=True,
                    help="Pad naar opgeslagen .keras of .h5 model")
parser.add_argument("--backbone", "-b", choices=["mobilenet", "mobilenetv3large",
                                                 "efficientnet", "efficientnetb2",
                                                 "efficientnetb3", "efficientnetv2s"],
                    required=True,
                    help="Backbone die gebruikt is bij training")
parser.add_argument("--img_size", "-s", type=int, default=224,
                    help="Image size (default: 224)")
parser.add_argument("--batch_size", "-bs", type=int, default=16,
                    help="Batch size (default: 16)")
parser.add_argument("--out_path", "-o", type=str, default="confusion_matrix.png",
                    help="Output bestand voor de confusion matrix figuur")
args = parser.parse_args()

# ----------------- Preprocess functie -----------------
def get_preprocess_fn(name):
    if name == "mobilenet":
        from tensorflow.keras.applications import mobilenet_v2
        return mobilenet_v2.preprocess_input
    if name == "mobilenetv3large":
        from tensorflow.keras.applications import mobilenet_v3
        return mobilenet_v3.preprocess_input
    if name in ["efficientnet", "efficientnetb2", "efficientnetb3"]:
        from tensorflow.keras.applications import efficientnet
        return efficientnet.preprocess_input
    from tensorflow.keras.applications import efficientnet_v2
    return efficientnet_v2.preprocess_input

preprocess_fn = get_preprocess_fn(args.backbone)

# ----------------- Generator -----------------
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    preprocessing_function=preprocess_fn,
    validation_split=0.2
)

val_gen = datagen.flow_from_directory(
    args.data_dir,
    target_size=(args.img_size, args.img_size),
    batch_size=args.batch_size,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

# ----------------- Model laden -----------------
print(f"Model laden vanaf: {args.model_path}")
model = tf.keras.models.load_model(args.model_path)

# ----------------- Voorspellingen -----------------
print("Voorspellingen maken...")
preds = model.predict(val_gen, verbose=1)
y_pred = np.argmax(preds, axis=1)
y_true = val_gen.classes
class_labels = list(val_gen.class_indices.keys())

# ----------------- Confusion matrix -----------------
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
fig, ax = plt.subplots(figsize=(12, 10))
disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, values_format="d")
plt.title("Confusion Matrix")
plt.savefig(args.out_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"Confusion matrix opgeslagen als: {args.out_path}")

# ----------------- Rapport -----------------
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_labels))
