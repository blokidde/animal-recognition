import os
import time
import argparse
import numpy as np
import tensorflow as tf

from train_animal_classifier import (
    build_data_generators,
    get_base_model,
    build_model,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train MobileNetV2 and EfficientNetB0 models and compare performance"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Root directory with one subfolder per class",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=224,
        help="Square size to which images are resized (default: 224)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for training (default: 16)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of epochs (default: 30)",
    )
    return parser.parse_args()


def train_model(model_name, args):
    """Train a model and return training time and saved path."""
    train_gen, val_gen = build_data_generators(
        args.data_dir, args.img_size, args.batch_size
    )
    base = get_base_model(model_name, args.img_size)
    model = build_model(base, train_gen.num_classes)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    log_dir = os.path.join("logs", model_name)
    os.makedirs(log_dir, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3
        ),
        tf.keras.callbacks.TensorBoard(log_dir=log_dir),
    ]

    start = time.time()
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks,
    )
    train_time = time.time() - start

    out_path = f"custom_{model_name}_model.h5"
    model.save(out_path)

    final_acc = None
    if "val_accuracy" in history.history:
        final_acc = history.history["val_accuracy"][-1]

    return train_time, out_path, final_acc


def evaluate_model(model_path, img_size):
    print(f"Evaluating {model_path}...")
    size_bytes = os.path.getsize(model_path)

    start_load = time.time()
    model = tf.keras.models.load_model(model_path)
    load_time = time.time() - start_load

    dummy = np.random.rand(1, img_size, img_size, 3).astype(np.float32)
    start_inf = time.time()
    model.predict(dummy, verbose=0)
    infer_time = time.time() - start_inf

    return size_bytes, load_time, infer_time


def main():
    args = parse_args()

    results = []
    for model_name in ["mobilenet", "efficientnet"]:
        print(f"\n=== Training {model_name} ===")
        try:
            train_time, path, final_acc = train_model(model_name, args)
            size_bytes, load_time, infer_time = evaluate_model(path, args.img_size)
            results.append(
                {
                    "model": model_name,
                    "train_time": train_time,
                    "size_mb": size_bytes / (1024 * 1024),
                    "load_time": load_time,
                    "infer_time": infer_time,
                    "val_acc": final_acc,
                }
            )
        except Exception as exc:
            print(f"Training {model_name} failed: {exc}")

    print("\nResultaten:")
    header = f"{'Model':<12}{'Train (s)':>12}{'Size (MB)':>12}{'Load (s)':>12}{'Infer (s)':>12}{'Val acc':>10}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['model']:<12}{r['train_time']:>12.2f}{r['size_mb']:>12.2f}{r['load_time']:>12.2f}{r['infer_time']:>12.4f}{(r['val_acc'] if r['val_acc'] is not None else 'n/a'):>10}"
        )


if __name__ == "__main__":
    main()
