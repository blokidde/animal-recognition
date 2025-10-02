import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image

# === Instellingen ===
model_path = "/mnt/e/MachineLearning/new_animal_model/tested_models/working_model_1_10_25/custom_mobilenetv3large_model.keras"
images_dir = "/mnt/e/datasets/filtered/living/Meles_meles"  # map met dassenfoto's
train_dir = "/mnt/e/MachineLearning/new_animal_model/animal_photos/simple_images"  # originele trainingsmap
img_size = (224, 224)

# === Model laden ===
model = tf.keras.models.load_model(model_path)

# === Klassen automatisch ophalen (alfabetische volgorde van mappen in train_dir) ===
class_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
print("Klassenvolgorde:", class_names)

# Index van "badger" bepalen
if "badger" not in class_names:
    raise ValueError("'badger' staat niet in je trainingsklassen! Klassen gevonden: " + str(class_names))
badger_idx = class_names.index("badger")

# === Functie om kans op das te berekenen ===
def predict_badger(img_path):
    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array, verbose=0)
    prob_badger = preds[0][badger_idx]
    return prob_badger

# === Loop door alle foto's in de map ===
for fname in os.listdir(images_dir):
    if fname.lower().endswith((".jpg", ".jpeg", ".png")):
        fpath = os.path.join(images_dir, fname)
        prob = predict_badger(fpath)
        print(f"{fname} → kans op dassenfoto: {prob:.2f}")
