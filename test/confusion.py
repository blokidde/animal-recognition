import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# 1. Pad naar je dataset en model
data_dir = "/mnt/e/MachineLearning/new_animal_model/animal_photos/simple_images"
model_path = "/mnt/e/MachineLearning/new_animal_model/tested_models/working_model_29_08_25/custom_mobilenetv3large_model.h5"  # pas aan als je andere naam hebt

# 2. Zelfde generator als bij training
img_size = 224
batch_size = 16

datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, validation_split=0.2)

val_gen = datagen.flow_from_directory(
    data_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="categorical",
    subset="validation",
    shuffle=False  # heel belangrijk! anders kloppen y_true indices niet
)

# 3. Model laden
model = tf.keras.models.load_model(model_path)

# 4. Voorspellingen maken
preds = model.predict(val_gen, verbose=1)
y_pred = np.argmax(preds, axis=1)
y_true = val_gen.classes
class_labels = list(val_gen.class_indices.keys())

# 5. Confusion matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
fig, ax = plt.subplots(figsize=(12, 10))
disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix")
plt.savefig("/mnt/e/Machinelearning/new_animal_model", dpi=300, bbox_inches="tight")
plt.show()

# 6. Rapport printen
print(classification_report(y_true, y_pred, target_names=class_labels))
