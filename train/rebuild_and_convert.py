#!/usr/bin/env python3
from keras.models import load_model   # let op: 'keras', niet 'tf.keras'
import tensorflow as tf

h5_path = "/mnt/e/MachineLearning/new_animal_model/tested_models/working_model_29_08_25/custom_mobilenetv3large_model.h5"
out_dir = "/mnt/e/MachineLearning/new_animal_model/tested_models/working_model_29_08_25/custom_mobilenetv3large_model"

print("🔄 H5 laden via standalone Keras (safe_mode=False)...")
model = load_model(h5_path, compile=False, safe_mode=False)

print("💾 Wegschrijven als SavedModel...")
tf.saved_model.save(model, out_dir)
print("✅ Klaar.")
