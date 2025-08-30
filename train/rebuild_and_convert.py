#!/usr/bin/env python3
import tensorflow as tf
from train_animal_classifier import get_base_model, build_model  # <-- juiste bestandsnaam gebruiken

# === Paden ===
h5_model_path = "/mnt/e/MachineLearning/new_animal_model/tested_models/working_model_29_08_25/custom_mobilenetv3large_model.h5"
saved_model_dir = "/mnt/e/MachineLearning/new_animal_model/tested_models/working_model_29_08_25/custom_mobilenetv3large_model"

# === Bouw model na ===
base, _ = get_base_model("mobilenetv3large", 224)
model = build_model(base, num_classes=20)  # pas 20 aan als je minder/meer klassen hebt

# === Laad gewichten ===
print("🔄 Gewichten laden uit H5...")
model.load_weights(h5_model_path)

# === Opslaan in SavedModel formaat ===
print("💾 Opslaan naar SavedModel...")
tf.saved_model.save(model, saved_model_dir)

print("✅ Klaar! Model staat nu als SavedModel in:", saved_model_dir)
