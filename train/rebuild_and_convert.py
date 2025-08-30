#!/usr/bin/env python3
import tensorflow as tf
from train_animal_classifier import get_base_model, build_model

# === Paden ===
h5_model_path = "/mnt/e/MachineLearning/new_animal_model/tested_models/working_model_29_08_25/custom_mobilenetv3large_model.h5"
saved_model_dir = "/mnt/e/MachineLearning/new_animal_model/tested_models/working_model_29_08_25/custom_mobilenetv3large_model"

# === Bouw hetzelfde model na ===
num_classes = 20  # pas dit aan naar het aantal klassen in jouw dataset
base, _ = get_base_model("mobilenetv3large", 224)
model = build_model(base, num_classes)

# === Alleen de gewichten laden ===
print("🔄 Laden van gewichten...")
model.load_weights(h5_model_path)

# === Opslaan in SavedModel formaat ===
print(f"💾 Opslaan naar {saved_model_dir} ...")
tf.saved_model.save(model, saved_model_dir)

print("✅ Klaar! Model staat nu in SavedModel formaat.")
