#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import sys
import math
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    TensorBoard,
    ModelCheckpoint,
)

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a custom classifier on an image dataset with transfer learning"
    )
    parser.add_argument(
        "--data_dir", "-d", type=str, required=True,
        help="Root directory with one subfolder per class"
    )
    parser.add_argument(
        "--model", "-m",
        choices=["mobilenet", "mobilenetv3large", "efficientnet", "efficientnetb2", "efficientnetb3", "efficientnetv2s"],
        default="efficientnet",
        help="Backbone to use (default: efficientnet)"
    )
    parser.add_argument(
        "--img_size", "-s", type=int, default=224,
        help="Square image size (default: 224)"
    )
    parser.add_argument(
        "--batch_size", "-b", type=int, default=16,
        help="Batch size (default: 16)"
    )
    parser.add_argument(
        "--epochs", "-e", type=int, default=30,
        help="Total epochs (default: 30)"
    )
    parser.add_argument(
        "--augment", action="store_true",
        help="Enable mild data augmentation"
    )
    parser.add_argument(
        "--val_split", type=float, default=0.2,
        help="Validation split for ImageDataGenerator (default: 0.2)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed used by generators (default: 42)"
    )
    parser.add_argument(
        "--log_dir", type=str, default="logs",
        help="TensorBoard log directory (default: logs)"
    )
    parser.add_argument(
        "--output_prefix", type=str, default=None,
        help="Optional prefix for output filenames (default uses model name)"
    )
    parser.add_argument(
        "--mixed_precision", action="store_true",
        help="Enable mixed float16 precision (GPU recommended)"
    )
    parser.add_argument(
        "--num_workers", type=int, default=None,
        help="Number of data loading workers for model.fit (default: auto)"
    )
    return parser.parse_args()


def enable_mixed_precision(flag: bool):
    if not flag:
        return
    try:
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy("mixed_float16")
        print("[Info] Mixed precision policy set to 'mixed_float16'.")
    except Exception as e:
        print(f"[Warn] Could not enable mixed precision: {e}")


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


def build_data_generators(data_dir, img_size, batch_size, preprocess_fn, enable_aug, val_split, seed):
    if enable_aug:
        datagen = ImageDataGenerator(
            preprocessing_function=preprocess_fn,
            validation_split=val_split,
            rotation_range=15,
            zoom_range=0.2,
            horizontal_flip=True
        )
    else:
        datagen = ImageDataGenerator(
            preprocessing_function=preprocess_fn,
            validation_split=val_split
        )

    train_gen = datagen.flow_from_directory(
        data_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="categorical",
        subset="training",
        shuffle=True,
        seed=seed
    )
    val_gen = datagen.flow_from_directory(
        data_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
        shuffle=False,  # belangrijk voor consistente evaluatie
        seed=seed
    )
    return train_gen, val_gen


def get_base_model(name, img_size):
    # Standaard "tail_unfreeze" per backbone
    if name == "mobilenet":
        base = tf.keras.applications.MobileNetV2(
            weights="imagenet", include_top=False,
            input_shape=(img_size, img_size, 3)
        )
        tail_unfreeze = 30
    elif name == "mobilenetv3large":
        base = tf.keras.applications.MobileNetV3Large(
            weights="imagenet", include_top=False,
            input_shape=(img_size, img_size, 3)
        )
        tail_unfreeze = 30
    elif name == "efficientnet":
        base = tf.keras.applications.EfficientNetB0(
            weights="imagenet", include_top=False,
            input_shape=(img_size, img_size, 3)
        )
        tail_unfreeze = 40
    elif name == "efficientnetb2":
        base = tf.keras.applications.EfficientNetB2(
            weights="imagenet", include_top=False,
            input_shape=(img_size, img_size, 3)
        )
        tail_unfreeze = 50
    elif name == "efficientnetb3":
        base = tf.keras.applications.EfficientNetB3(
            weights="imagenet", include_top=False,
            input_shape=(img_size, img_size, 3)
        )
        tail_unfreeze = 50
    else:  # efficientnetv2s
        base = tf.keras.applications.EfficientNetV2S(
            weights="imagenet", include_top=False,
            input_shape=(img_size, img_size, 3)
        )
        tail_unfreeze = 60

    base.trainable = False  # fase 1: backbone bevroren
    return base, tail_unfreeze


def build_model(base_model, num_classes):
    inputs = layers.Input(shape=base_model.input_shape[1:])
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(
        256, activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(0.01)
    )(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(
        128, activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(0.01)
    )(x)
    # Bij mixed precision forceren we float32 op de output voor stabiliteit
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        dtype="float32"  # belangrijk i.c.m. mixed_float16
    )(x)
    return models.Model(inputs, outputs)


def count_trainable_params(model) -> int:
    return int(
        sum(
            tf.size(v) for v in model.trainable_variables
        )
    )


def main():
    args = parse_args()

    # Eventueel mixed precision
    enable_mixed_precision(args.mixed_precision)

    # Voor TensorBoard
    os.makedirs(args.log_dir, exist_ok=True)

    preprocess_fn = get_preprocess_fn(args.model)
    train_gen, val_gen = build_data_generators(
        args.data_dir, args.img_size, args.batch_size, preprocess_fn,
        args.augment, args.val_split, args.seed
    )
    num_classes = train_gen.num_classes
    print("Classes:", train_gen.class_indices)

    # Sla class mapping op voor inference
    with open("class_indices.json", "w") as f:
        json.dump(train_gen.class_indices, f, indent=2)

    # Backbone + head
    base, tail_unfreeze = get_base_model(args.model, args.img_size)
    model = build_model(base, num_classes)

    # Metrics
    topk = min(5, num_classes)
    topk_metric = tf.keras.metrics.TopKCategoricalAccuracy(k=topk, name=f"top{topk}")

    # Callbacks
    ckpt_path = f"best_{args.model}.keras" if not args.output_prefix else f"{args.output_prefix}_best_{args.model}.keras"
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
        TensorBoard(log_dir=args.log_dir),
        ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True)
    ]

    # Fit configuratie (voor Windows meestal geen multiprocessing)
    if args.num_workers is None:
        workers = max(1, (os.cpu_count() or 2) // 2)
    else:
        workers = args.num_workers
    use_mp = (os.name != "nt")  # veilig default

    # ===== Fase 1: train alleen de classifier head =====
    warmup_epochs = max(5, args.epochs // 3)
    print(f"\n[Phase 1] Training classifier head (backbone frozen) for {warmup_epochs} epochs...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy", topk_metric],
    )
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=warmup_epochs,
        callbacks=callbacks,
        verbose=1
    )

    # ===== Fase 2: unfreeze tail en fijn-afstellen =====
    remaining_epochs = max(0, args.epochs - warmup_epochs)
    print(f"\n[Phase 2] Unfreezing last {tail_unfreeze} layers of the backbone...")
    base.trainable = True
    if tail_unfreeze > 0:
        for layer in base.layers[:-tail_unfreeze]:
            layer.trainable = False

    # Logging hoeveel lagen trainable zijn
    trainable_layers = sum(1 for l in model.layers if l.trainable)
    print(f"[Info] Trainable layers after unfreeze: {trainable_layers}")
    print(f"[Info] Trainable params: {count_trainable_params(model):,}")

    if remaining_epochs > 0:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
            metrics=["accuracy", topk_metric],
        )
        model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=remaining_epochs,
            callbacks=callbacks,
            verbose=1
        )
    else:
        print("[Info] No epochs left for fine-tuning phase; skipping Phase 2.")

    # ===== Opslaan (eerst .keras, daarna SavedModel) =====
    prefix = args.output_prefix if args.output_prefix else f"custom_{args.model}_model"
    out_file = f"{prefix}.keras"          # single file → Keras v3 format
    out_dir = f"{prefix}_saved"           # directory → SavedModel

    # Keras-bestand (.keras)
    model.save(out_file)
    print(f"Keras modelbestand geschreven naar: {out_file}")

    # SavedModel (map)
    model.export(out_dir)
    print(f"SavedModel geschreven naar map: {out_dir}/")

    print(f"Best checkpoint (volgens val_loss) staat ook op: {ckpt_path}")
    print("Klaar")



if __name__ == "__main__":
    # Snellere cuDNN init bij eerste run kan wat logs geven; dat is normaal.
    # Optioneel: minder TF-logs
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
    main()
