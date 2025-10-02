#!/usr/bin/env python3
# convert_to_tflite.py
import argparse, os, sys, pathlib
import numpy as np
import tensorflow as tf

# ---------------------------------------------
# Preprocessing helpers (optioneel "inbakken")
# ---------------------------------------------
def build_preprocessing_layer(kind: str, img_size: int):
    """
    - 'none'            : geen preprocessing in model (je voert zelf 0..1 of al-gepreprocessed in)
    - 'rescale_1_255'   : x/255.0  (verwacht uint8 0..255, wordt 0..1)
    - 'mobilenet_v3'    : tf.keras.applications.mobilenet_v3.preprocess_input ([-1,1])
    - 'efficientnet'    : tf.keras.applications.efficientnet.preprocess_input
    """
    Input = tf.keras.layers.Input(shape=(img_size, img_size, 3), dtype=tf.uint8, name="raw_input_uint8")
    x = tf.keras.layers.Lambda(lambda t: tf.cast(t, tf.float32), name="to_float32")(Input)

    if kind == "rescale_1_255":
        x = tf.keras.layers.Rescaling(1./255., name="rescale_1_255")(x)
    elif kind == "mobilenet_v3":
        from tensorflow.keras.applications import mobilenet_v3
        x = tf.keras.layers.Lambda(mobilenet_v3.preprocess_input, name="mobilenet_v3_preproc")(x)
    elif kind == "efficientnet":
        from tensorflow.keras.applications import efficientnet
        x = tf.keras.layers.Lambda(efficientnet.preprocess_input, name="efficientnet_preproc")(x)
    elif kind == "none":
        pass
    else:
        raise ValueError(f"Onbekende preprocessing '{kind}'.")
    return Input, x

# ---------------------------------------------
# Representative dataset voor INT8 kalibratie
# ---------------------------------------------
def make_representative_gen(rep_dir: str, img_size: int, baked_preproc: bool, preproc_kind: str, limit: int = 500):
    img_ext = {".jpg", ".jpeg", ".png", ".bmp"}
    paths = []
    for root, _, files in os.walk(rep_dir):
        for f in files:
            if pathlib.Path(f).suffix.lower() in img_ext:
                paths.append(os.path.join(root, f))
    paths = paths[:limit]

    def load_and_resize(p):
        raw = tf.io.read_file(p)
        img = tf.io.decode_image(raw, channels=3, expand_animations=False)
        img = tf.image.resize(img, (img_size, img_size), antialias=True)
        return img

    def generator():
        for p in paths:
            img = load_and_resize(p)
            if baked_preproc:
                img_uint8 = tf.cast(tf.clip_by_value(img, 0, 255), tf.uint8).numpy()
                yield [np.expand_dims(img_uint8, axis=0)]
            else:
                img_f32 = tf.cast(img, tf.float32) / 255.0
                if preproc_kind == "mobilenet_v3":
                    from tensorflow.keras.applications import mobilenet_v3
                    img_f32 = mobilenet_v3.preprocess_input(img_f32)
                elif preproc_kind == "efficientnet":
                    from tensorflow.keras.applications import efficientnet
                    img_f32 = efficientnet.preprocess_input(img_f32)
                yield [np.expand_dims(img_f32.numpy(), axis=0)]
    return generator

# ---------------------------------------------
# Output-pad helper
# ---------------------------------------------
def make_output_path(model_path: str, quant: str, int8_io_uint8: bool) -> str:
    base = pathlib.Path(model_path)
    out_dir = base.parent / "tflite"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = ""
    if quant == "int8" and int8_io_uint8:
        suffix = "_int8_uint8io"
    elif quant == "int8":
        suffix = "_int8"
    elif quant == "fp16":
        suffix = "_fp16"
    else:
        suffix = "_float32"
    name = base.stem + suffix + ".tflite"
    return str(out_dir / name)

# ---------------------------------------------
# Main
# ---------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Converteer .keras/.h5 naar TFLite met optionele preprocessing en kwantisatie.")
    ap.add_argument("--model", required=True, help="Pad naar .keras/.h5 model")
    ap.add_argument("--out", required=False, help="Optioneel: pad naar .tflite output (anders automatisch in ./tflite/)")
    ap.add_argument("--img_size", type=int, default=224, help="Input resolutie (vierkant)")
    ap.add_argument("--bake_preprocessing",
                    choices=["none", "rescale_1_255", "mobilenet_v3", "efficientnet"],
                    default="rescale_1_255",
                    help="Preprocessing in model bakken (aanbevolen).")
    ap.add_argument("--quant", choices=["none", "fp16", "int8"], default="int8",
                    help="Kwantisatie: none/fp16/int8")
    ap.add_argument("--rep_data_dir", default=None, help="Map met representatieve afbeeldingen voor INT8 kalibratie")
    ap.add_argument("--int8_io_uint8", action="store_true",
                    help="Bij INT8: forceer uint8 in-/output (0..255).")
    ap.add_argument("--limit_rep_images", type=int, default=500, help="Max # representatieve afbeeldingen")
    args = ap.parse_args()

    model_path = args.model
    img_size   = args.img_size
    bake_kind  = args.bake_preprocessing
    quant      = args.quant

    if not os.path.exists(model_path):
        print(f"Model niet gevonden: {model_path}", file=sys.stderr)
        sys.exit(1)

    # Bepaal output-pad (en maak ./tflite/)
    out_path = args.out or make_output_path(model_path, quant, args.int8_io_uint8)

    print(f"[Info] Laden van basismodel: {model_path}")
    base_model = tf.keras.models.load_model(model_path)

    # Preprocessing inbakken (optioneel)
    baked_preproc = bake_kind != "none"
    if baked_preproc:
        print(f"[Info] Preprocessing in model bakken: {bake_kind}")
        raw_in, pre_x = build_preprocessing_layer(bake_kind, img_size)
        y = base_model(pre_x)
        model = tf.keras.Model(inputs=raw_in, outputs=y, name="model_with_preproc")
    else:
        print("[Info] Geen preprocessing ingebakken; model verwacht reeds gepreprocesseerde float32.")
        model = base_model

    # --- Converter ---
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # >>> Belangrijk: sta TF Select (Flex) ops toe om errors als 'tf.Relu not a flex op' op te lossen
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    converter.experimental_enable_resource_variables = True

    if quant == "none":
        print("[Info] Converter: geen kwantisatie (float32 TFLite).")
    elif quant == "fp16":
        print("[Info] Converter: FP16 kwantisatie.")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif quant == "int8":
        print("[Info] Converter: INT8 kwantisatie.")
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if not args.rep_data_dir or not os.path.isdir(args.rep_data_dir):
            print("[Fout] INT8 vereist --rep_data_dir met representatieve afbeeldingen.", file=sys.stderr)
            sys.exit(2)
        rep_gen = make_representative_gen(
            rep_dir=args.rep_data_dir,
            img_size=img_size,
            baked_preproc=baked_preproc,
            preproc_kind=bake_kind,
            limit=args.limit_rep_images
        )
        converter.representative_dataset = rep_gen
        if args.int8_io_uint8:
            converter.inference_input_type  = tf.uint8
            converter.inference_output_type = tf.uint8
            print("[Info] INT8 met uint8 IO (0..255).")
        else:
            converter.inference_input_type  = tf.int8
            converter.inference_output_type = tf.int8
            print("[Info] INT8 met int8 IO (-128..127).")
    else:
        raise ValueError("Onbekende kwantisatie-optie")

    # Convert + wegschrijven
    print(f"[Info] Start conversie → {out_path}")
    tflite_bytes = converter.convert()
    pathlib.Path(os.path.dirname(out_path) or ".").mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(tflite_bytes)
    print(f"[Klaar] TFLite model opgeslagen: {out_path}")

    # IO-samenvatting
    try:
        interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
        interpreter.allocate_tensors()
        ids_in  = interpreter.get_input_details()
        ids_out = interpreter.get_output_details()
        print("\n[Samenvatting IO]")
        for i, d in enumerate(ids_in):
            print(f"  Input[{i}]  name={d['name']} shape={d['shape']} dtype={d['dtype']}")
        for i, d in enumerate(ids_out):
            print(f"  Output[{i}] name={d['name']} shape={d['shape']} dtype={d['dtype']}")
    except Exception as e:
        print(f"[Waarschuwing] Kon IO-samenvatting niet tonen: {e}")

if __name__ == "__main__":
    main()
