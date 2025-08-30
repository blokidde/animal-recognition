#!/usr/bin/env python3
import argparse
import json
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a custom classifier on an image dataset with transfer learning"
    )
    parser.add_argument("--data_dir", "-d", type=str, required=True,
                        help="Root directory with one subfolder per class")
    parser.add_argument("--model", "-m",
                        choices=["mobilenet", "mobilenetv3large", "efficientnet", "efficientnetb2", "efficientnetb3", "efficientnetv2s"],
                        default="efficientnet",
                        help="Backbone to use (default: efficientnet)")
    parser.add_argument("--img_size", "-s", type=int, default=224,
                        help="Square image size (default: 224)")
    parser.add_argument("--batch_size", "-b", type=int, default=16,
                        help="Batch size (default: 16)")
    parser.add_argument("--epochs", "-e", type=int, default=30,
                        help="Total epochs (default: 30)")
    parser.add_argument("--augment", action="store_true",
                        help="Enable mild data augmentation")
    return parser.parse_args()

def get_preprocess_fn(name):
    if name == "mobilenet":
        from tensorflow.keras.applications import mobilenet_v2
        return mobilenet_v2.preprocess_input
    if name == "mobilenetv3large":
        from tensorflow.keras.applications import mobilenet_v3
        return mobilenet_v3.preprocess_input
    if name in ["efficientnet", "efficientnetb2", "efficientnetb3"]:
        from tensorflow.keras.applications import efficientnet
        return efficientnet.preprocess_input
    # efficientnetv2s
    from tensorflow.keras.applications import efficientnet_v2
    return efficientnet_v2.preprocess_input

def build_data_generators(data_dir, img_size, batch_size, preprocess_fn, enable_aug):
    if enable_aug:
        datagen = ImageDataGenerator(
            preprocessing_function=preprocess_fn,
            validation_split=0.2,
            rotation_range=15,
            zoom_range=0.2,
            horizontal_flip=True
        )
    else:
        datagen = ImageDataGenerator(
            preprocessing_function=preprocess_fn,
            validation_split=0.2
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
        base = tf.keras.applications.MobileNetV2(weights="imagenet", include_top=False,
                                                 input_shape=(img_size, img_size, 3))
        tail_unfreeze = 30
    elif name == "mobilenetv3large":
        base = tf.keras.applications.MobileNetV3Large(weights="imagenet", include_top=False,
                                                      input_shape=(img_size, img_size, 3))
        tail_unfreeze = 30
    elif name == "efficientnet":
        base = tf.keras.applications.EfficientNetB0(weights="imagenet", include_top=False,
                                                    input_shape=(img_size, img_size, 3))
        tail_unfreeze = 40
    elif name == "efficientnetb2":
        base = tf.keras.applications.EfficientNetB2(weights="imagenet", include_top=False,
                                                    input_shape=(img_size, img_size, 3))
        tail_unfreeze = 50
    elif name == "efficientnetb3":
        base = tf.keras.applications.EfficientNetB3(weights="imagenet", include_top=False,
                                                    input_shape=(img_size, img_size, 3))
        tail_unfreeze = 50
    else:  # efficientnetv2s
        base = tf.keras.applications.EfficientNetV2S(weights="imagenet", include_top=False,
                                                     input_shape=(img_size, img_size, 3))
        tail_unfreeze = 60
    base.trainable = False  # Phase 1: backbone frozen
    return base, tail_unfreeze

def build_model(base_model, num_classes):
    inputs = layers.Input(shape=base_model.input_shape[1:])
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax",
                           kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    return models.Model(inputs, outputs)

def main():
    args = parse_args()

    preprocess_fn = get_preprocess_fn(args.model)
    train_gen, val_gen = build_data_generators(
        args.data_dir, args.img_size, args.batch_size, preprocess_fn, args.augment
    )
    num_classes = train_gen.num_classes
    print("Classes:", train_gen.class_indices)

    # Save mapping for inference later
    with open("class_indices.json", "w") as f:
        json.dump(train_gen.class_indices, f, indent=2)

    base, tail_unfreeze = get_base_model(args.model, args.img_size)
    model = build_model(base, num_classes)

    top5 = tf.keras.metrics.TopKCategoricalAccuracy(k=5, name="top5")

    os.makedirs("logs", exist_ok=True)
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
        TensorBoard(log_dir="logs"),
    ]

    # ===== Phase 1: train head only =====
    warmup_epochs = max(5, args.epochs // 3)
    print(f"\n[Phase 1] Training classifier head with backbone frozen for {warmup_epochs} epochs...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy", top5],
    )
    model.fit(train_gen, validation_data=val_gen, epochs=warmup_epochs, callbacks=callbacks)

    # ===== Phase 2: unfreeze the tail and fine-tune =====
    print(f"\n[Phase 2] Unfreezing last {tail_unfreeze} layers of the backbone...")
    base.trainable = True
    for layer in base.layers[:-tail_unfreeze]:
        layer.trainable = False

    remaining_epochs = max(0, args.epochs - warmup_epochs)
    if remaining_epochs > 0:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
            metrics=["accuracy", top5],
        )
        model.fit(train_gen, validation_data=val_gen, epochs=remaining_epochs, callbacks=callbacks)

    out_path_dir = f"custom_{args.model}_model"
    model.save(out_path_dir, save_format="tf")
    print(f"Model opgeslagen in SavedModel formaat: {out_path_dir}/")

    out_path = f"custom_{args.model}_model.h5"
    model.save(out_path)
    print(f"Model opgeslagen als {out_path}")

if __name__ == "__main__":
    main()
