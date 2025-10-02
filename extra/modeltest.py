import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

model_path = "/mnt/e/MachineLearning/new_animal_model/tested_models/working_model_31_08_25/custom_mobilenetv3large_model.keras"
img_size = (224, 224)

# Laden van model
model = tf.keras.models.load_model(model_path)

# Klassen in de juiste volgorde
class_names = ['badger', 'beaver', 'brown_bear', 'european_polecat', 'fallow_deer',
               'fox', 'hare', 'lynx', 'mallard', 'mouflon', 'pheasant',
               'pine_marten', 'rabbit', 'raccoon', 'raccoon_dog', 'red_deer',
               'roe_deer', 'stone_marten', 'wild_boar', 'wolf']

# Kies een voorbeeldfoto (pas aan naar jouw testmap)
img_path = "/mnt/e/datasets/filtered/living/Meles_meles/263738404_0.jpg"

img = image.load_img(img_path, target_size=img_size)
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

preds = model.predict(img_array, verbose=0)[0]

print(f"Voorspellingen voor {os.path.basename(img_path)}:\n")
for cname, prob in zip(class_names, preds):
    print(f"{cname:15s}: {prob:.4f}")
