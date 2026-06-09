#!/usr/bin/env python3
import argparse
from pathlib import Path
import tensorflow as tf

def main() -> None:
    parser = argparse.ArgumentParser(description="Laad een legacy H5-model en schrijf het weg als SavedModel.")
    parser.add_argument(
        "--h5_path",
        default="tested_models/working_model_29_08_25/custom_mobilenetv3large_model.h5",
        help="Pad naar het H5-model.",
    )
    parser.add_argument(
        "--out_dir",
        default="tested_models/working_model_29_08_25/custom_mobilenetv3large_model",
        help="Uitvoermap voor SavedModel.",
    )
    args = parser.parse_args()

    h5_path = Path(args.h5_path)
    out_dir = Path(args.out_dir)

    print("Laden met TF 2.12 CPU (legacy H5 loader is toleranter)...")
    model = tf.keras.models.load_model(h5_path, compile=False)

    print("Wegschrijven als SavedModel...")
    tf.saved_model.save(model, out_dir)

    print("Klaar. SavedModel in:", out_dir)


if __name__ == "__main__":
    main()
