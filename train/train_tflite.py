#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Train classifier + export to TFLite (fp32/fp16/int8)."
    )
    p.add_argument("--data_dir", "-d", required=True, type=str,
                   help="Root dir met 1 submap per klasse.")
    p.add_argument("--model", "-m",
                   choices=["mobilenet", "mobilenetv3large", "efficientnet",
                            "efficientnetb2", "efficientnetb3", "efficientnetv2s"],
                   default="efficientnet")
    p.add_argument("--img_size", "-s", type=int, default=224)
    p.add_argument("--batch_size", "-b", type=int, default=16)
    p.add_argument("--epochs", "-e", type=int, default=30)
    p.add_argument("--augment", action="store_true")
    p.add_argument("--val_split", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_dir", type=str, default="logs")
    p.add_argument("--output_prefix", type=str, default=None)
    p.add_argument("--mixed_precision", action="store_true",
                   help="Set global policy float16 (alleen trainen; TFLite export blijft los).")
    # TFLite export opties
    p.add_argument("--export_fp32", action="store_true", help="Exporteer on-quantized .tflite")
    p.add_argument("--export_fp16", action="store_true", help="Exporteer fp16 .tflite")
    p.add_argument("--export_int8_dynamic", action="store_true", help="Exporteer int8 dynamic range .tflite")
    p.add_argument("--export_int8_full", action="store_true", help="Exporteer full-int8 .tflite (vereist --rep_data_dir)")
    p.add_argument("--rep_data_dir", type=str, default=None,
                   help="Pad naar representatieve afbeeldingen (vereist voor --export_int8_full).")
    return p.parse_args()

# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------
def enable_mixed_precision(flag: bool):
    if not flag:
        return
    try:
        from tensorflow.keras import mixed_precision
        mixed_precision.set_global_policy("mixed_float16")
        print("[Info] Mixed precision aan (float16).")
    except Exception as e:
        print(f"[Warn] Mixed precision niet gelukt: {e}")

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
            rotation_range=15, zoom_range=0.2, horizontal_flip=True
        )
    else:
        datagen = ImageDataGenerator(
            preprocessing_function=preprocess_fn,
            validation_split=val_split
        )
    train_gen = datagen.flow_from_directory(
        data_dir, target_size=(img_size, img_size),
        batch_size=batch_size, class_mode="categorical",
        subset="training", shuffle=True, seed=seed
    )
    val_gen = datagen.flow_from_directory(
        data_dir, target_size=(img_size, img_size),
        batch_size=batch_size, class_mode="categorical",
        subset="validation", shuffle=False, seed=seed
    )
    return train_gen, val_gen

def get_base_model(name, img_size):
    if name == "mobilenet":
        base = tf.keras.applications.MobileNetV2(
            weights="imagenet", include_top=False, input_shape=(img_size, img_size, 3)
        )
        tail_unfreeze = 30
    elif name == "mobilenetv3large":
        base = tf.keras.applications.MobileNetV3Large(
            weights="imagenet", include_top=False, input_shape=(img_size, img_size, 3)
        )
        tail_unfreeze = 30
    elif name == "efficientnet":
        base = tf.keras.applications.EfficientNetB0(
            weights="imagenet", include_top=False, input_shape=(img_size, img_size, 3)
        )
        tail_unfreeze = 40
    elif name == "efficientnetb2":
        base = tf.keras.applications.EfficientNetB2(
            weights="imagenet", include_top=False, input_shape=(img_size, img_size, 3)
        )
        tail_unfreeze = 50
    elif name == "efficientnetb3":
        base = tf.keras.applications.EfficientNetB3(
            weights="imagenet", include_top=False, input_shape=(img_size, img_size, 3)
        )
        tail_unfreeze = 50
    else:  # efficientnetv2s
        base = tf.keras.applications.EfficientNetV2S(
            weights="imagenet", include_top=False, input_shape=(img_size, img_size, 3)
        )
        tail_unfreeze = 60
    base.trainable = False
    return base, tail_unfreeze

def build_model(base_model, num_classes):
    inputs = layers.Input(shape=base_model.input_shape[1:])
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(256, activation="relu",
                     kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)
    return models.Model(inputs, outputs)

# ------------------------------------------------------------
# TFLite converters
# ------------------------------------------------------------
def export_tflite_fp32(keras_model, out_path):
    conv = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    tflite_bytes = conv.convert()
    Path(out_path).write_bytes(tflite_bytes)
    print(f"[TFLite] FP32 → {out_path}")

def export_tflite_fp16(keras_model, out_path):
    conv = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.target_spec.supported_types = [tf.float16]
    tflite_bytes = conv.convert()
    Path(out_path).write_bytes(tflite_bytes)
    print(f"[TFLite] FP16 → {out_path}")

def export_tflite_int8_dynamic(keras_model, out_path):
    conv = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]  # zonder rep. dataset = dynamic range
    tflite_bytes = conv.convert()
    Path(out_path).write_bytes(tflite_bytes)
    print(f"[TFLite] INT8 dynamic range → {out_path}")

def make_representative_dataset(rep_data_dir, img_size, preprocess_fn, max_images=500):
    # generator die preprocessed float32 batches van size 1 levert
    img_paths = []
    for root, _, files in os.walk(rep_data_dir):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                img_paths.append(os.path.join(root, f))
    # Beperk voor snelheid
    img_paths = img_paths[:max_images]

    def gen():
        for p in img_paths:
            img = load_img(p, target_size=(img_size, img_size))
            arr = img_to_array(img)
            arr = preprocess_fn(arr)  # << BELANGRIJK: zelfde preprocessing als training
            arr = np.expand_dims(arr.astype("float32"), axis=0)
            yield [arr]
    return gen

def export_tflite_int8_full(keras_model, out_path, rep_data_dir, img_size, preprocess_fn):
    if rep_data_dir is None or not os.path.isdir(rep_data_dir):
        raise ValueError("--rep_data_dir is vereist voor full-int8 en moet een map zijn.")
    conv = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = make_representative_dataset(rep_data_dir, img_size, preprocess_fn)
    # Volledige int8 pipeline (alle operators int8):
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    tflite_bytes = conv.convert()
    Path(out_path).write_bytes(tflite_bytes)
    print(f"[TFLite] FULL-INT8 → {out_path}")

# ------------------------------------------------------------
# Train + Export
# ------------------------------------------------------------
def main():
    args = parse_args()
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
    enable_mixed_precision(args.mixed_precision)
    os.makedirs(args.log_dir, exist_ok=True)

    preprocess_fn = get_preprocess_fn(args.model)
    train_gen, val_gen = build_data_generators(
        args.data_dir, args.img_size, args.batch_size,
        preprocess_fn, args.augment, args.val_split, args.seed
    )
    num_classes = train_gen.num_classes
    print("Classes:", train_gen.class_indices)
    with open("class_indices.json", "w") as f:
        json.dump(train_gen.class_indices, f, indent=2)

    base, tail_unfreeze = get_base_model(args.model, args.img_size)
    model = build_model(base, num_classes)

    topk = min(5, num_classes)
    topk_metric = tf.keras.metrics.TopKCategoricalAccuracy(k=topk, name=f"top{topk}")

    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, ModelCheckpoint
    ckpt_path = (args.output_prefix + "_best.keras") if args.output_prefix else f"best_{args.model}.keras"
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
        TensorBoard(log_dir=args.log_dir),
        ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True)
    ]

    # Phase 1: head
    warmup_epochs = max(5, args.epochs // 3)
    print(f"\n[Phase 1] Head trainen ({warmup_epochs} epochs) ...")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy", topk_metric],
    )
    model.fit(train_gen, validation_data=val_gen, epochs=warmup_epochs, callbacks=callbacks, verbose=1)

    # Phase 2: finetune tail
    remaining = max(0, args.epochs - warmup_epochs)
    print(f"\n[Phase 2] Unfreeze laatste {tail_unfreeze} lagen ...")
    base.trainable = True
    for layer in base.layers[:-tail_unfreeze]:
        layer.trainable = False

    if remaining > 0:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(5e-5),
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
            metrics=["accuracy", topk_metric],
        )
        model.fit(train_gen, validation_data=val_gen, epochs=remaining, callbacks=callbacks, verbose=1)

    # Opslaan
    prefix = args.output_prefix if args.output_prefix else f"custom_{args.model}_model"
    keras_path = f"{prefix}.keras"
    saved_dir = f"{prefix}_saved"
    model.save(keras_path)
    model.export(saved_dir)
    print(f"[Save] .keras → {keras_path}")
    print(f"[Save] SavedModel → {saved_dir}/")
    print(f"[Info] Best checkpoint stond op: {ckpt_path}")

    # Voor TFLite export: laad de beste weights (zekerheid)
    best_model = tf.keras.models.load_model(ckpt_path)

    # Export padbasis
    tfl_dir = Path(f"{prefix}_tflite")
    tfl_dir.mkdir(parents=True, exist_ok=True)

    # Exports op basis van vlaggen
    if args.export_fp32:
        export_tflite_fp32(best_model, str(tfl_dir / f"{prefix}_fp32.tflite"))
    if args.export_fp16:
        export_tflite_fp16(best_model, str(tfl_dir / f"{prefix}_fp16.tflite"))
    if args.export_int8_dynamic:
        export_tflite_int8_dynamic(best_model, str(tfl_dir / f"{prefix}_int8_dynamic.tflite"))
    if args.export_int8_full:
        export_tflite_int8_full(
            best_model, str(tfl_dir / f"{prefix}_int8_full.tflite"),
            args.rep_data_dir, args.img_size, preprocess_fn
        )

    print("\nKlaar.")

if __name__ == "__main__":
    main()
