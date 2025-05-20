import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import os

# === Instellingen ===
MODEL_PATH = "custom_mobilenet_model.h5"
IMG_PATH = "test_pics/rabbit_test.jpg"
IMG_SIZE = 224

# === Class names automatisch ophalen uit trainingsstructuur ===
DATA_DIR = "/mnt/c/Users/jurriaan/MachineLearning/animal_photos/simple_images"
temp_gen = ImageDataGenerator().flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=1,
    class_mode="categorical"
)
CLASS_NAMES = list(temp_gen.class_indices.keys())

# === Model laden ===
model = tf.keras.models.load_model(MODEL_PATH)

# === Afbeelding voorbereiden ===
img = load_img(IMG_PATH, target_size=(IMG_SIZE, IMG_SIZE))
img_array = img_to_array(img)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

# === Voorspelling maken ===
predictions = model.predict(img_array)
predicted_index = np.argmax(predictions)
predicted_class = CLASS_NAMES[predicted_index]

# === Output tonen ===
print(f"Afbeelding: {os.path.basename(IMG_PATH)}")
print(f"🔍 Voorspelde klasse: {predicted_class}")
print("📊 Verdeling:")
for i, class_name in enumerate(CLASS_NAMES):
    print(f"   {class_name:>12}: {predictions[0][i]:.4f}")
