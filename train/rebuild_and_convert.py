#!/usr/bin/env python3
import tensorflow as tf

h5_path = "/mnt/e/MachineLearning/new_animal_model/tested_models/working_model_29_08_25/custom_mobilenetv3large_model.h5"
out_dir = "/mnt/e/MachineLearning/new_animal_model/tested_models/working_model_29_08_25/custom_mobilenetv3large_model"

print("🔄 Laden met TF 2.12 CPU (legacy H5 loader is toleranter)…")
model = tf.keras.models.load_model(h5_path, compile=False)

print("💾 Wegschrijven als SavedModel…")
tf.saved_model.save(model, out_dir)

print("✅ Klaar. SavedModel in:", out_dir)
