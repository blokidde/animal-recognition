#!/usr/bin/env python3
import argparse
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import regularizers, losses

def main() -> None:
    parser = argparse.ArgumentParser(description="Converteer een H5-model naar SavedModel.")
    parser.add_argument(
        "--h5_model_path",
        default="tested_models/working_model_29_08_25/custom_mobilenetv3large_model.h5",
    )
    parser.add_argument(
        "--saved_model_dir",
        default="tested_models/working_model_29_08_25/custom_mobilenetv3large_model",
    )
    args = parser.parse_args()

    h5_model_path = Path(args.h5_model_path)
    saved_model_dir = Path(args.saved_model_dir)

    print(f"Laden van H5 model: {h5_model_path}")

    # Custom objects die gebruikt zijn tijdens training
    custom_objs = {
        "CategoricalCrossentropy": losses.CategoricalCrossentropy,
        "l2": regularizers.l2,
    }

    # Model laden (compile=False voorkomt problemen met optimizer/loss)
    model = tf.keras.models.load_model(h5_model_path, custom_objects=custom_objs, compile=False)

    # Opslaan in TensorFlow SavedModel formaat
    print(f"Opslaan naar map: {saved_model_dir}")
    model.save(saved_model_dir, save_format="tf")

    print("Conversie gelukt. Je model staat nu in SavedModel formaat.")


if __name__ == "__main__":
    main()
