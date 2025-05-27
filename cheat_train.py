import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Pad naar je dataset
DATA_DIR = "/mnt/c/Users/jurriaan/MachineLearning/animal_photos/simple_images"
IMG_SIZE = 224
BATCH_SIZE = 64

# Data generator ZONDER augmentatie
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.0)

train_gen = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

# Neem slechts 1 batch om op te trainen
images, labels = next(train_gen)

# Bouw een klein model (MobileNetV2)
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_model.trainable = False  # bevries convolutionele lagen

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(labels.shape[1], activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train op slechts 1 batch, meerdere keren
model.fit(images, labels, epochs=20, verbose=2)

# Evaluatie
loss, acc = model.evaluate(images, labels, verbose=0)
print(f"\n✅ Final training accuracy on 1 batch: {acc:.4f}")
