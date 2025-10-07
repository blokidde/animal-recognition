# python3 test.py --yolo_model yolo11n.pt --cls_model /mnt/e/MachineLearning/new_animal_model/tested_models/working_model_1_10_25/custom_mobilenetv3large_model.keras --data_dir /mnt/e/MachineLearning/new_animal_model/animal_photos/simple_images --source /mnt/e/MachineLearning/new_animal_model/zwijnen_close.mp4 --save /mnt/e/MachineLearning/new_animal_model/zwijnen_close_out.mp4 --img_size 224 --conf 0.25 --iou 0.5 --margin 0.1 --min_side 64 --preprocess mobilenet_v3
#!/usr/bin/env python3
import argparse, os, numpy as np, cv2, tensorflow as tf
from ultralytics import YOLO

# -------------------- Args --------------------
p = argparse.ArgumentParser(description="YOLO detectie + crops classificeren")
p.add_argument("--yolo_model", type=str, default="yolo11n.pt", help="Ultralytics YOLO model (.pt)")
p.add_argument("--cls_model",  type=str, required=True, help="Pad naar .keras of .h5 classifier")
p.add_argument("--data_dir",   type=str, required=True, help="Root met 1 subfolder per klasse (om class_names op te halen)")
p.add_argument("--source",     type=str, default="0", help="Afbeelding/videopad of '0' voor webcam")
p.add_argument("--img_size",   type=int, default=224, help="Classifier input size")
p.add_argument("--conf",       type=float, default=0.25, help="YOLO conf threshold")
p.add_argument("--iou",        type=float, default=0.5, help="YOLO NMS IoU")
p.add_argument("--max_det",    type=int, default=20, help="Max detections per frame")
p.add_argument("--min_side",   type=int, default=64, help="Minimale korte zijde ROI (px) om te classificeren")
p.add_argument("--margin",     type=float, default=0.10, help="Extra margerand rond ROI (relatief)")
p.add_argument("--preprocess", choices=["auto","rescale","mobilenet_v3","none"], default="auto",
              help="Preprocess voor classifier")
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
from tensorflow.keras.preprocessing.image import ImageDataGenerator
tmp = ImageDataGenerator().flow_from_directory(
    args.data_dir, target_size=(args.img_size, args.img_size),
    batch_size=1, class_mode="categorical"
)
CLASS_NAMES = list(tmp.class_indices.keys())

# -------------------- Load models --------------------
yolo = YOLO(args.yolo_model)
cls_model = tf.keras.models.load_model(args.cls_model)

def has_rescaling_layer(m):
    return any(isinstance(l, tf.keras.layers.Rescaling) for l in m.layers)

def get_preprocess_kind():
    if args.preprocess != "auto":
        return args.preprocess
    return "rescale" if has_rescaling_layer(cls_model) else "rescale"  # conservatief
    # Tip: zet hier "mobilenet_v3" als je met Keras MobileNetV3 preprocess trainde.

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
    # BGR -> RGB
    roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    if args.letterbox:
        roi = letterbox_to_square(roi, args.img_size)
    else:
        roi = cv2.resize(roi, (args.img_size, args.img_size), interpolation=cv2.INTER_LINEAR)
    roi = roi.astype(np.float32)
    if PREP == "rescale":
        roi = roi / 255.0
    elif PREP == "mobilenet_v3":
        roi = mv3_pre(roi)
    # 'none' -> laat zoals het is
    return np.expand_dims(roi, 0)

def classify_crop(roi_bgr):
    batch = prep_roi_for_classifier(roi_bgr)
    pred = cls_model.predict(batch, verbose=0)[0]
    idx = int(np.argmax(pred))
    return CLASS_NAMES[idx], float(pred[idx])

def expand_box(xyxy, w, h, mratio):
    x1, y1, x2, y2 = map(int, xyxy)
    bw, bh = x2 - x1, y2 - y1
    mx, my = int(bw * mratio), int(bh * mratio)
    nx1, ny1 = max(0, x1 - mx), max(0, y1 - my)
    nx2, ny2 = min(w, x2 + mx), min(h, y2 + my)
    return nx1, ny1, nx2, ny2

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
        # sla te kleine crops over
        if min(x2 - x1, y2 - y1) < args.min_side:
            cv2.rectangle(frame, (x1,y1), (x2,y2), (128,128,128), 1)
            cv2.putText(frame, f"small ({conf_det:.2f})", (x1, max(0, y1-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128,128,128), 1, cv2.LINE_AA)
            continue

        ex1, ey1, ex2, ey2 = expand_box((x1,y1,x2,y2), w, h, args.margin)
        crop = frame[ey1:ey2, ex1:ex2]

        try:
            species, conf_cls = classify_crop(crop)

            # --- Threshold check ---
            if conf_cls < 0.75:
                label = f"low conf ({species} {conf_cls:.2f})"
                color = (128, 128, 128)  # grijs
            else:
                label = f"{species} {conf_cls:.2f} | det {conf_det:.2f}"
                color = (0, 200, 0)  # groen
        except Exception as e:
            label = f"cls err"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
        # label achtergrond
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

    return frame

# -------------------- Run (image/webcam/video) --------------------
src = args.source
is_cam = (src == "0" or src == "1" or src == "2")
if is_cam:
    cap = cv2.VideoCapture(int(src))
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = None
    else:
        out = None

    while True:
        ok, frame = cap.read()
        if not ok: break
        vis = handle_frame(frame)
        if args.save:
            if out is None:
                h, w = vis.shape[:2]
                out = cv2.VideoWriter(args.save, fourcc, 25, (w, h))
            out.write(vis)
        # cv2.imshow("YOLO -> Classify", vis)
        # if cv2.waitKey(1) & 0xFF == ord('q'): break
    cap.release()
    if out: out.release()
    cv2.destroyAllWindows()

else:
    # afbeelding of videobestand
    if os.path.isfile(src) and os.path.splitext(src)[1].lower() in [".jpg",".jpeg",".png",".bmp",".webp"]:
        img = cv2.imread(src)
        vis = handle_frame(img)
        if args.save:
            os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
            cv2.imwrite(args.save, vis)
            print(f"Saved: {args.save}")
        # else:
            # cv2.imshow("YOLO -> Classify", vis)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
    else:
        # video
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise SystemExit(f"Kon bron niet openen: {src}")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = None
        while True:
            ok, frame = cap.read()
            if not ok: break
            vis = handle_frame(frame)
            if args.save:
                if out is None:
                    h, w = vis.shape[:2]
                    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
                    out = cv2.VideoWriter(args.save, fourcc, 25, (w, h))
                out.write(vis)
            # cv2.imshow("YOLO -> Classify", vis)
            # if cv2.waitKey(1) & 0xFF == ord('q'): break
        cap.release()
        if out: out.release()
        cv2.destroyAllWindows()
