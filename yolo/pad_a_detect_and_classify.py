import argparse
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

CLASSES = ['badger', 'beaver', 'fallow_deer', 'fox', 'hare', 'lynx', 'mouflon', 
           'pheasant', 'rabbit', 'raccoon', 'red_deer', 'roe_deer', 'wild_boar', 'wolf']

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

def preprocess_crop(image_path: str, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """Preprocess crop for classification."""
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def classify_crops(crop_paths: List[str], model: tf.keras.Model) -> List[Tuple[str, float]]:
    """Batch classify crops."""
    if not crop_paths:
        return []
    
    batch_images = []
    for crop_path in crop_paths:
        try:
            img = preprocess_crop(crop_path)
            batch_images.append(img[0])
        except Exception as e:
            print(f"Error preprocessing {crop_path}: {e}")
            batch_images.append(np.zeros((224, 224, 3)))
    
    batch_images = np.array(batch_images)
    predictions = model.predict(batch_images, verbose=0)
    
    results = []
    for pred in predictions:
        class_idx = np.argmax(pred)
        confidence = float(pred[class_idx])
        species = CLASSES[class_idx]
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
    parser.add_argument('--images', default='/mnt/e/MachineLearning/new_animal_model/animal_photos/simple_images')
    parser.add_argument('--yolo_model', default='yolov8n.pt')
    parser.add_argument('--classifier', default='/home/jurriaan/animalrec/models/animals_model_efficient.h5')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--device', default='0')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.images):
        raise FileNotFoundError(f"Images directory not found: {args.images}")
    
    # Load models
    print("Loading YOLO model...")
    yolo = YOLO(args.yolo_model)
    
    print("Loading classifier...")
    classifier = load_classifier(args.classifier)
    
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
            classifications = classify_crops(crop_paths, classifier)
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
