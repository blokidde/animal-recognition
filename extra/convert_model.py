#!/usr/bin/env python3
import tensorflow as tf
from tensorflow.keras import regularizers, losses

# === Pad naar je H5 model ===
h5_model_path = "/mnt/e/MachineLearning/new_animal_model/tested_models/working_model_29_08_25/custom_mobilenetv3large_model.h5"

# === Waar je SavedModel map moet komen ===
saved_model_dir = "/mnt/e/MachineLearning/new_animal_model/tested_models/working_model_29_08_25/custom_mobilenetv3large_model"

print(f"🔄 Laden van H5 model: {h5_model_path}")

# Custom objects die gebruikt zijn tijdens training
custom_objs = {
    "CategoricalCrossentropy": losses.CategoricalCrossentropy,
    "l2": regularizers.l2,
}

# Model laden (compile=False voorkomt problemen met optimizer/loss)
model = tf.keras.models.load_model(h5_model_path, custom_objects=custom_objs, compile=False)

# Opslaan in TensorFlow SavedModel formaat
print(f"💾 Opslaan naar map: {saved_model_dir}")
model.save(saved_model_dir, save_format="tf")

print("✅ Conversie gelukt! Je model staat nu in SavedModel formaat.")
