import os, json, hashlib, argparse, time
from pathlib import Path
from google_images_search import GoogleImagesSearch
from PIL import Image
import imagehash
import tensorflow as tf
import numpy as np

# ---------- CLI ---------- #
ap = argparse.ArgumentParser()
ap.add_argument("--cfg",  default="species.json",
                help="JSON met soorten & zoektermen")
ap.add_argument("--num",  type=int, default=250,
                help="Min. aantal foto's per soort")
ap.add_argument("--out",  default="dataset_raw", help="Hoofdmap")
ap.add_argument("--check", action="store_true",
                help="Gebruik snelle MobileNet-check")
args = ap.parse_args()

# ---------- Keys ---------- #
from dotenv import load_dotenv; load_dotenv()
gis = GoogleImagesSearch(os.getenv("GCS_API_KEY1"), os.getenv("GCS_CX1"))

# ---------- Optionele MobileNet verifiëring ---------- #
if args.check:
    print("🔎 Initialiseer MobileNet-verificatie …")
    verifier = tf.keras.applications.MobileNetV2(weights="imagenet")
    preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
    decode    = tf.keras.applications.imagenet_utils.decode_predictions

# ---------- Hulpfuncties ---------- #
def is_duplicate(img_path, seen_hashes, thr=1):
    try:
        h = imagehash.phash(Image.open(img_path))
        if any(abs(h - sh) <= thr for sh in seen_hashes):
            return True
        seen_hashes.add(h)
    except Exception:
        return True  # kapotte of niet-beeld bestanden overslaan
    return False

def is_correct_species(img_path, keywords):
    """Kijk of top-3 ImageNet labels 1 van de keywords bevat"""
    if not args.check:
        return True
    try:
        img = Image.open(img_path).convert("RGB").resize((224,224))
        x = preprocess(np.array(img)[None, ...].astype("float32"))
        preds = verifier.predict(x, verbose=0)
        labels = [l[1].lower() for l in decode(preds, top=3)[0]]
        return any(kw in label for kw in keywords for label in labels)
    except Exception:
        return False

# ---------- Main ---------- #
with open(args.cfg) as f:
    species = json.load(f)

for sp, terms in species.items():
    out_dir = Path(args.out)/sp
    out_dir.mkdir(parents=True, exist_ok=True)
    seen = set()

    downloaded = len(list(out_dir.glob("*.jpg")))
    term_idx   = 0
    while downloaded < args.num and term_idx < len(terms):
        q = terms[term_idx]
        term_idx += 1
        print(f"\n▶️  {sp}: query '{q}' …")
        search_params = dict(q=q, num=min(10,args.num-downloaded),
                             imgType="photo", fileType="jpg", safe="off")
        gis.search(search_params, path_to_dir=str(out_dir))

        for img_path in list(out_dir.glob("*.jpg"))[-10:]:  # alleen nieuwe files
            # dedup
            if is_duplicate(img_path, seen):
                img_path.unlink(missing_ok=True); continue
            # minimale resolutie
            if Image.open(img_path).size[0] < 400:
                img_path.unlink(missing_ok=True); continue
            # snelle MobileNet-check
            if not is_correct_species(img_path,
                                      [sp.replace('_',' '), *terms]):
                img_path.unlink(missing_ok=True); continue
            downloaded += 1
            print(f"   ✔︎  {downloaded}/{args.num}", end="\r")

    print(f"✅  {sp}: {downloaded} beelden in {out_dir}")
print("\n🎉  Klaar!")
