import argparse
import json
import os
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import tensorflow as tf
from ultralytics import YOLO
from tqdm import tqdm
import cv2
from typing import List, Tuple, Optional

DEFAULT_CLASS_INDICES = Path("train/class_indices.json")

def find_crop_paths_for_image(crops_root: str, image_stem: str) -> list[str]:
    """Zoek alle crops voor een bronafbeelding (ongeacht submap/klasse)."""
    root = Path(crops_root)
    if not root.exists():
        return []
    # Ultralytics cropnamen bevatten vaak de originele naam + index
    matches = []
    for sub in root.glob("*"):
        if sub.is_dir():
            for cp in sub.glob(f"{image_stem}*.jpg"):
                matches.append(str(cp))
    return sorted(matches)

def load_classifier(model_path: str) -> tf.keras.Model:
    """Load Keras classification model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Classifier model not found: {model_path}")
    return tf.keras.models.load_model(model_path)

def load_class_names(class_indices_path: str) -> list[str]:
    """Load model class names in prediction-index order."""
    path = Path(class_indices_path)
    if not path.exists():
        raise FileNotFoundError(f"Class mapping not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        class_indices = json.load(f)

    return [name for name, _ in sorted(class_indices.items(), key=lambda item: item[1])]

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
    if name == "efficientnetv2s":
        from tensorflow.keras.applications import efficientnet_v2
        return efficientnet_v2.preprocess_input
    return lambda arr: arr / 255.0

def preprocess_crop(
    image_path: str,
    preprocess_fn,
    target_size: Tuple[int, int] = (224, 224),
) -> np.ndarray:
    """Preprocess crop for classification."""
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img, dtype=np.float32)
    img_array = preprocess_fn(img_array)
    return np.expand_dims(img_array, axis=0)

def classify_crops(
    crop_paths: List[str],
    model: tf.keras.Model,
    class_names: list[str],
    preprocess_fn,
    img_size: int,
) -> List[Tuple[str, float]]:
    """Batch classify crops."""
    if not crop_paths:
        return []

    batch_images = []
    for crop_path in crop_paths:
        try:
            img = preprocess_crop(crop_path, preprocess_fn, (img_size, img_size))
            batch_images.append(img[0])
        except Exception as e:
            print(f"Error preprocessing {crop_path}: {e}")
            batch_images.append(np.zeros((img_size, img_size, 3)))
    
    batch_images = np.array(batch_images)
    predictions = model.predict(batch_images, verbose=0)
    
    results = []
    for pred in predictions:
        class_idx = np.argmax(pred)
        confidence = float(pred[class_idx])
        species = class_names[class_idx] if class_idx < len(class_names) else "unknown"
        results.append((species, confidence))
    
    return results

def parse_yolo_results(results_dir: str, image_name: str) -> List[dict]:
    """Parse YOLO detection results from txt files."""
    txt_file = os.path.join(results_dir, 'labels', f"{os.path.splitext(image_name)[0]}.txt")
    
    if not os.path.exists(txt_file):
        return []
    
    detections = []
    with open(txt_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:  # class, x, y, w, h, conf
                detection = {
                    'x_center_n': float(parts[1]),
                    'y_center_n': float(parts[2]),
                    'w_n': float(parts[3]),
                    'h_n': float(parts[4]),
                    'yolo_conf': float(parts[5])
                }
                detections.append(detection)
    
    return detections

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', default='animal_photos/simple_images')
    parser.add_argument('--yolo_model', default='animal_photos/yolov8n.pt')
    parser.add_argument('--classifier', default='train/best_mobilenetv3large.keras')
    parser.add_argument('--class_indices', default=str(DEFAULT_CLASS_INDICES))
    parser.add_argument(
        '--preprocess',
        choices=['rescale', 'mobilenet', 'mobilenetv3large', 'efficientnet', 'efficientnetb2', 'efficientnetb3', 'efficientnetv2s'],
        default='mobilenetv3large',
        help='Zelfde preprocessing/backbone als waarmee de classifier getraind is.'
    )
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--classifier_img_size', type=int, default=224)
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--device', default='cpu')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.images):
        raise FileNotFoundError(f"Images directory not found: {args.images}")
    
    # Load models
    print("Loading YOLO model...")
    yolo = YOLO(args.yolo_model)
    
    print("Loading classifier...")
    classifier = load_classifier(args.classifier)
    class_names = load_class_names(args.class_indices)
    preprocess_fn = get_preprocess_fn(args.preprocess)
    
    # Run YOLO detection
    print("Running YOLO detection...")
    results_dir = "yolo"
    yolo.predict(
        source=args.images,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        save_txt=True,
        save_conf=True,
        save_crop=True,
        project=results_dir,
        name="detect",
        exist_ok=True
    )
    
    # Process results
    detect_dir = os.path.join(results_dir, "detect")
    crops_root = os.path.join(detect_dir, "crops")
    
    results_data = []
    
    # Get all images
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(Path(args.images).glob(ext))
    
    print("Processing detections...")
    for image_file in tqdm(image_files):
        image_name = image_file.name
        detections = parse_yolo_results(detect_dir, image_name)
        
        if not detections:
            continue
        
        # Find corresponding crops
        base_name = os.path.splitext(image_name)[0]
        crop_paths = find_crop_paths_for_image(crops_root, base_name)
        
        # Classify crops
        if crop_paths:
            classifications = classify_crops(
                crop_paths,
                classifier,
                class_names,
                preprocess_fn,
                args.classifier_img_size,
            )
        else:
            classifications = []
        
        # Combine results
        for i, detection in enumerate(detections):
            if i < len(crop_paths) and i < len(classifications):
                species, species_conf = classifications[i]
                crop_path = os.path.relpath(crop_paths[i])
            else:
                species, species_conf = "unknown", 0.0
                crop_path = ""
            
            result = {
                'image_path': os.path.relpath(str(image_file)),
                'x_center_n': detection['x_center_n'],
                'y_center_n': detection['y_center_n'],
                'w_n': detection['w_n'],
                'h_n': detection['h_n'],
                'yolo_conf': detection['yolo_conf'],
                'species': species,
                'species_conf': species_conf,
                'crop_path': crop_path
            }
            results_data.append(result)
    
    # Save results
    df = pd.DataFrame(results_data)
    df.to_csv('results.csv', index=False)
    print(f"Saved {len(results_data)} detections to results.csv")

if __name__ == "__main__":
    main()
