#!/usr/bin/env python3
import argparse, os, numpy as np, cv2, tensorflow as tf
import tensorflow.lite as tflite
try:
    import ai_edge_litert as litert  # Optional: newer LiteRT interpreter
except Exception:
    litert = None
from ultralytics import YOLO

# -------------------- Args --------------------
p = argparse.ArgumentParser(description="YOLO detectie + TFLite-classificatie per crop")
p.add_argument("--yolo_model", type=str, default="yolo11n.pt", help="Ultralytics YOLO model (.pt)")
p.add_argument("--cls_model",  type=str, required=True, help="Pad naar .tflite classifier (FP16 of INT8)")
p.add_argument("--data_dir",   type=str, required=True, help="Root met 1 subfolder per klasse (om class_names op te halen)")
p.add_argument("--source",     type=str, default="0", help="Afbeelding/videopad of '0' voor webcam")
p.add_argument("--img_size",   type=int, default=224, help="Classifier input size")
p.add_argument("--conf",       type=float, default=0.25, help="YOLO conf threshold")
p.add_argument("--iou",        type=float, default=0.5, help="YOLO NMS IoU")
p.add_argument("--max_det",    type=int, default=20, help="Max detections per frame")
p.add_argument("--min_side",   type=int, default=64, help="Minimale korte zijde ROI (px) om te classificeren")
p.add_argument("--margin",     type=float, default=0.10, help="Extra margerand rond ROI (relatief)")
p.add_argument("--preprocess", choices=["auto","rescale","mobilenet_v3","none"], default="auto",
              help="Preprocess voor classifier (bij TFLite 'auto' = 'rescale')")
p.add_argument("--letterbox",  action="store_true", help="Behoud aspectratio met padding i.p.v. hard resize")
p.add_argument("--save",       type=str, default="", help="Opslaan naar bestand (image) of video (mp4)")
args = p.parse_args()

# -------------------- TF init --------------------
try:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass

# -------------------- Class names --------------------
# Let op: dit scant je hele data_dir (kan traag zijn). Later kun je dit vervangen door class_names.json.
from tensorflow.keras.preprocessing.image import ImageDataGenerator
tmp = ImageDataGenerator().flow_from_directory(
    args.data_dir, target_size=(args.img_size, args.img_size),
    batch_size=1, class_mode="categorical"
)
CLASS_NAMES = list(tmp.class_indices.keys())

# -------------------- Load models --------------------
yolo = YOLO(args.yolo_model)

# TFLite Interpreter (+ Flex delegate for Select TF ops)
if not args.cls_model.lower().endswith(".tflite"):
    raise SystemExit("Geef een .tflite model op bij --cls_model voor deze TFLite-versie.")

def _try_load_flex_delegate():
    # Try loading Flex delegate so models with Select TF ops (e.g., FlexMul) run
    loaders = []
    # Prefer LiteRT loader if available, otherwise fall back to TF's experimental loader
    if litert and hasattr(litert, "load_delegate"):
        loaders.append(litert.load_delegate)
    if hasattr(tf.lite, "experimental") and hasattr(tf.lite.experimental, "load_delegate"):
        loaders.append(tf.lite.experimental.load_delegate)

    # Candidate library names across platforms; current OS is Linux, so .so will likely work
    candidates = (
        "libtensorflowlite_flex.so",  # Linux
        "libtensorflowlite_flex.dylib",  # macOS
        "tensorflowlite_flex.dll",  # Windows (older naming)
        "libtensorflowlite_flex.dll",  # Windows (alt)
    )

    # Potential directories to look for the library
    tf_dir = os.path.dirname(tf.__file__)
    dirs = [
        os.environ.get("TFLITE_FLEX_DELEGATE_PATH", ""),
        tf_dir,
        os.path.join(tf_dir, "lite"),
    ]
    try:
        import tensorflow as _tf
        if hasattr(_tf, "sysconfig") and hasattr(_tf.sysconfig, "get_lib"):
            dirs.append(_tf.sysconfig.get_lib())
    except Exception:
        pass

    # Try direct names first, then absolute paths in known dirs
    from ctypes.util import find_library
    for load in loaders:
        # 1) direct names
        for lib in candidates:
            # If it's a bare name, attempt to locate with find_library to avoid creating invalid delegates
            if not os.path.isabs(lib):
                base = lib
                if base.startswith("lib") and base.endswith(('.so', '.dylib', '.dll')):
                    base = base[len("lib"):].split('.')[0]
                found = find_library(base)
                if found:
                    lib_to_load = found
                else:
                    # Skip attempting to load if not found to prevent partial delegate creation
                    continue
            else:
                lib_to_load = lib
            try:
                delegate = load(lib_to_load)
                print(f"Loaded Flex delegate: {lib_to_load}")
                return delegate
            except Exception:
                continue
        # 2) absolute paths in known dirs
        for d in filter(None, dirs):
            for lib in candidates:
                abs_path = os.path.join(d, lib)
                if os.path.exists(abs_path):
                    try:
                        delegate = load(abs_path)
                        print(f"Loaded Flex delegate: {abs_path}")
                        return delegate
                    except Exception:
                        continue
    return None

delegates = []
flex_delegate = _try_load_flex_delegate()
if flex_delegate is not None:
    delegates.append(flex_delegate)

###############################################
# Create interpreter with a few strategies
###############################################
last_err = None

# Prefer LiteRT Interpreter if available, else fallback to TF Lite Interpreter
InterpreterCls = litert.Interpreter if (litert and hasattr(litert, "Interpreter")) else tflite.Interpreter

# Strategy 1: Try with provided delegates
try:
    interpreter = InterpreterCls(model_path=args.cls_model, experimental_delegates=delegates or None)
    interpreter.allocate_tensors()
except Exception as e:
    last_err = e
    # Strategy 2: If TF exposes OpResolverType with SELECT_TF_OPS, try that
    try:
        op_type = getattr(getattr(tf.lite, "experimental", object()), "OpResolverType", None)
        if op_type is not None and hasattr(op_type, "BUILTIN_WITH_SELECT_TF_OPS"):
            interpreter = InterpreterCls(
                model_path=args.cls_model,
                experimental_op_resolver_type=op_type.BUILTIN_WITH_SELECT_TF_OPS,
                experimental_delegates=delegates or None,
            )
            interpreter.allocate_tensors()
        else:
            raise RuntimeError("OpResolverType with SELECT_TF_OPS not available")
    except Exception as e2:
        last_err = e2
        hint = (
            "Interpreter allocate_tensors() faalde. Tip: installeer 'ai-edge-litert' of voer met Flex delegate. "
            "Op Linux hoort 'libtensorflowlite_flex.so' aanwezig te zijn bij volledige TensorFlow. "
            "Anders installeer volledige TensorFlow (niet alleen tflite_runtime) of pas je model aan om Select TF ops te vermijden."
        )
        raise SystemExit(f"TFLite initialisatie mislukt: {last_err}\n{hint}")
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Bepaal I/O dtypes (FP16 modellen gebruiken meestal float32 input)
in_idx  = input_details[0]["index"]
in_dtype = input_details[0]["dtype"]          # np.float32 of np.int8
out_idx = output_details[0]["index"]
out_dtype = output_details[0]["dtype"]        # vaak np.float32 / np.int8

# Voor INT8 modellen: nodig voor zero-point/scale
in_scale, in_zp = (input_details[0].get("quantization", (0.0, 0)) or (0.0, 0))
out_scale, out_zp = (output_details[0].get("quantization", (0.0, 0)) or (0.0, 0))

def get_preprocess_kind():
    # We kunnen bij TFLite de layers niet introspecteren → 'auto' = 'rescale' (conservatief)
    if args.preprocess == "auto":
        return "rescale"
    return args.preprocess

PREP = get_preprocess_kind()
if PREP == "mobilenet_v3":
    from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as mv3_pre

def letterbox_to_square(img_rgb, new_size):
    h, w = img_rgb.shape[:2]
    scale = min(new_size / h, new_size / w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((new_size, new_size, 3), dtype=resized.dtype)
    top = (new_size - nh) // 2
    left = (new_size - nw) // 2
    canvas[top:top+nh, left:left+nw] = resized
    return canvas

def prep_roi_for_classifier(roi_bgr):
    roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    if args.letterbox:
        roi = letterbox_to_square(roi, args.img_size)
    else:
        roi = cv2.resize(roi, (args.img_size, args.img_size), interpolation=cv2.INTER_LINEAR)

    # Basis float beeld
    roi = roi.astype(np.float32)

    # Preprocess
    if PREP == "rescale":
        roi = roi / 255.0
    elif PREP == "mobilenet_v3":
        roi = mv3_pre(roi)
    # 'none' laat waarden in [0..255] -> maak ze alsnog float
    roi = np.expand_dims(roi, 0)  # (1,H,W,3)

    # Pas dtype aan voor TFLite input
    if in_dtype == np.int8:
        # kwantiseren naar int8
        if PREP != "rescale":
            # zorg dat values ~[0..1] zitten vóór kwantisatie
            roi = np.clip(roi, 0.0, 1.0)
        if in_scale and in_scale > 0:
            roi_q = (roi / in_scale + in_zp).round().astype(np.int8)
        else:
            # fallback (zou zelden nodig moeten zijn)
            roi_q = (roi * 255.0 - 128.0).clip(-128,127).astype(np.int8)
        return roi_q
    else:
        # float32 input (ook voor FP16 modellen)
        return roi.astype(np.float32)

def classify_crop(roi_bgr):
    batch = prep_roi_for_classifier(roi_bgr)
    interpreter.set_tensor(in_idx, batch)
    interpreter.invoke()
    pred = interpreter.get_tensor(out_idx)[0]

    # De-kwantiseer output indien nodig
    if out_dtype == np.int8 and out_scale and out_scale > 0:
        pred = (pred.astype(np.float32) - out_zp) * out_scale

    idx = int(np.argmax(pred))
    return CLASS_NAMES[idx], float(pred[idx])

def expand_box(xyxy, w, h, mratio):
    x1, y1, x2, y2 = map(int, xyxy)
    bw, bh = x2 - x1, y2 - y1
    mx, my = int(bw * mratio), int(bh * mratio)
    nx1, ny1 = max(0, x1 - mx), max(0, y1 - my)
    nx2, ny2 = min(w, x2 + mx), min(h, y2 + my)
    return nx1, ny1, nx2, ny2

# Warm-up: bouw kernels en graph
_ = yolo.predict(np.zeros((64,64,3), dtype=np.uint8))
dummy = np.zeros((1, args.img_size, args.img_size, 3), dtype=np.float32)
if in_dtype == np.int8 and in_scale and in_scale > 0:
    dummy_q = (dummy / in_scale + in_zp).round().astype(np.int8)
    interpreter.set_tensor(in_idx, dummy_q)
else:
    interpreter.set_tensor(in_idx, dummy)
interpreter.invoke()

def handle_frame(frame):
    h, w = frame.shape[:2]
    results = yolo.predict(frame, conf=args.conf, iou=args.iou, max_det=args.max_det, verbose=False)
    if not results:
        return frame
    res = results[0]
    if res.boxes is None or len(res.boxes) == 0:
        return frame

    for b in res.boxes:
        x1, y1, x2, y2 = map(int, b.xyxy.cpu().numpy()[0])
        conf_det = float(b.conf.cpu().numpy()[0])

        if min(x2 - x1, y2 - y1) < args.min_side:
            cv2.rectangle(frame, (x1,y1), (x2,y2), (128,128,128), 1)
            cv2.putText(frame, f"small ({conf_det:.2f})", (x1, max(0, y1-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128,128,128), 1, cv2.LINE_AA)
            continue

        ex1, ey1, ex2, ey2 = expand_box((x1,y1,x2,y2), w, h, args.margin)
        crop = frame[ey1:ey2, ex1:ex2]

        try:
            species, conf_cls = classify_crop(crop)
            label = f"{species} {conf_cls:.2f} | det {conf_det:.2f}"
            color = (0, 200, 0)
        except Exception:
            label = f"cls err"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

    return frame

# -------------------- Run (image/webcam/video) --------------------
src = args.source
is_cam = (src == "0" or src == "1" or src == "2")

if is_cam:
    cap = cv2.VideoCapture(int(src))
    out = None
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    while True:
        ok, frame = cap.read()
        if not ok: break
        vis = handle_frame(frame)
        if args.save:
            if out is None:
                h, w = vis.shape[:2]
                fps = cap.get(cv2.CAP_PROP_FPS) or 25
                os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
                out = cv2.VideoWriter(args.save, fourcc, fps, (w, h))
            out.write(vis)
    cap.release()
    if out: out.release()

else:
    # afbeelding of videobestand
    if os.path.isfile(src) and os.path.splitext(src)[1].lower() in [".jpg",".jpeg",".png",".bmp",".webp"]:
        img = cv2.imread(src)
        vis = handle_frame(img)
        if args.save:
            os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
            cv2.imwrite(args.save, vis)
            print(f"Saved: {args.save}")
    else:
        # video
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise SystemExit(f"Kon bron niet openen: {src}")
        out = None
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        while True:
            ok, frame = cap.read()
            if not ok: break
            vis = handle_frame(frame)
            if args.save:
                if out is None:
                    h, w = vis.shape[:2]
                    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
                    out = cv2.VideoWriter(args.save, fourcc, fps, (w, h))
                out.write(vis)
        cap.release()
        if out: out.release()
