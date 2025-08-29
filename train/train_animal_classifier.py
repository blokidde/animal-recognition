#!/usr/bin/env python3
import argparse
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a custom classifier on your 9-class image dataset"
    )
    parser.add_argument(
        "--data_dir", "-d", type=str, required=True,
        help="Root directory with one subfolder per class"
    )
    parser.add_argument(
        "--model", "-m", choices=["mobilenet", "efficientnet"],
        default="efficientnet",
        help="Base model to use (default: efficientnet)"
    )
    parser.add_argument(
        "--img_size", "-s", type=int, default=224,
        help="Square size to which images are resized (default: 224)"
    )
    parser.add_argument(
        "--batch_size", "-b", type=int, default=16,
        help="Batch size for training (default: 16)"
    )
    parser.add_argument(
        "--epochs", "-e", type=int, default=30,
        help="Maximum number of epochs (default: 30)"
    )
    return parser.parse_args()

def build_data_generators(data_dir, img_size, batch_size):
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )
    train_gen = datagen.flow_from_directory(
        data_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="categorical",
        subset="training"
    )
    val_gen = datagen.flow_from_directory(
        data_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation"
    )
    return train_gen, val_gen

def get_base_model(name, img_size):
    if name == "mobilenet":
        base = tf.keras.applications.MobileNetV2(
            weights="imagenet", include_top=False,
            input_shape=(img_size, img_size, 3)
        )
    else:
        base = tf.keras.applications.EfficientNetB0(
            weights="imagenet", include_top=False,
            input_shape=(img_size, img_size, 3)
        )
    # Freeze alle lagen behalve de laatste 20
    base.trainable = True
    for layer in base.layers[:-20]:
        layer.trainable = False
    return base

def build_model(base_model, num_classes):
    inputs = layers.Input(shape=base_model.input_shape[1:])
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs, outputs)

def main():
    args = parse_args()

    # Data
    train_gen, val_gen = build_data_generators(
        args.data_dir, args.img_size, args.batch_size
    )
    num_classes = train_gen.num_classes

    # Model
    base = get_base_model(args.model, args.img_size)
    model = build_model(base, num_classes)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Callbacks
    os.makedirs("logs", exist_ok=True)
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
        TensorBoard(log_dir="logs")
    ]

    # Trainen
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks
    )

    # Opslaan
    out_path = f"custom_{args.model}_model.h5"
    model.save(out_path)
    print(f"Model opgeslagen als {out_path}")

if __name__ == "__main__":
    main()
