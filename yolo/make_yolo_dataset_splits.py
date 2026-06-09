import os
import shutil
import random
import argparse
import json
from pathlib import Path
from typing import List
import yaml

DEFAULT_SOURCE = Path("animal_photos/simple_images")
DEFAULT_TARGET = Path("datasets/animals")
DEFAULT_CLASS_INDICES = Path("train/class_indices.json")

def create_directory_structure(base_path: str) -> None:
    """Create YOLO dataset directory structure."""
    dirs = [
        'images/train',
        'images/val', 
        'labels/train',
        'labels/val'
    ]
    
    for dir_path in dirs:
        full_path = os.path.join(base_path, dir_path)
        os.makedirs(full_path, exist_ok=True)

def get_class_images(source_dir: str, class_name: str) -> List[str]:
    """Get all images for a specific class."""
    class_dir = os.path.join(source_dir, class_name)
    if not os.path.exists(class_dir):
        print(f"Warning: Class directory not found: {class_dir}")
        return []
    
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG', '*.BMP']:
        image_files.extend(Path(class_dir).glob(ext))
    
    return [str(f) for f in image_files]

def load_classes(source_dir: str, class_indices_path: str | None = None) -> list[str]:
    """Load class names from class_indices.json or from source subdirectories."""
    if class_indices_path:
        path = Path(class_indices_path)
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                class_indices = json.load(f)
            return [name for name, _ in sorted(class_indices.items(), key=lambda item: item[1])]

    return sorted(p.name for p in Path(source_dir).iterdir() if p.is_dir())

def split_and_copy_images(source_dir: str, target_dir: str, classes: list[str], train_ratio: float = 0.8) -> None:
    """Split images into train/val and copy to YOLO structure."""
    random.seed(42)
    
    train_images_dir = os.path.join(target_dir, 'images', 'train')
    val_images_dir = os.path.join(target_dir, 'images', 'val')
    
    total_train = 0
    total_val = 0
    
    for class_name in classes:
        print(f"Processing class: {class_name}")
        
        images = get_class_images(source_dir, class_name)
        if not images:
            continue
        
        random.shuffle(images)
        
        split_idx = int(len(images) * train_ratio)
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        
        # Copy train images
        for img_path in train_images:
            filename = os.path.basename(img_path)
            dest_path = os.path.join(train_images_dir, filename)
            shutil.copy2(img_path, dest_path)
        
        # Copy val images
        for img_path in val_images:
            filename = os.path.basename(img_path)
            dest_path = os.path.join(val_images_dir, filename)
            shutil.copy2(img_path, dest_path)
        
        total_train += len(train_images)
        total_val += len(val_images)
        
        print(f"  Train: {len(train_images)}, Val: {len(val_images)}")
    
    print(f"\nTotal - Train: {total_train}, Val: {total_val}")

def create_data_yaml(target_dir: str, classes: list[str]) -> None:
    """Create data.yaml file for YOLO training."""
    data = {
        'path': '.',
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(classes),
        'names': classes
    }
    
    yaml_path = os.path.join(target_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    
    print(f"Created data.yaml at {yaml_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default=str(DEFAULT_SOURCE))
    parser.add_argument('--target', default=str(DEFAULT_TARGET))
    parser.add_argument('--class_indices', default=str(DEFAULT_CLASS_INDICES))
    parser.add_argument('--train_ratio', type=float, default=0.8)
    
    args = parser.parse_args()
    
    if not os.path.exists(args.source):
        print(f"Error: Source directory not found: {args.source}")
        print("Please ensure the source directory exists with class subdirectories")
        return
    
    print(f"Creating YOLO dataset structure in: {args.target}")
    classes = load_classes(args.source, args.class_indices)
    if not classes:
        raise ValueError(f"Geen klassen gevonden in {args.source}")
    
    # Create directory structure
    create_directory_structure(args.target)
    
    # Split and copy images
    split_and_copy_images(args.source, args.target, classes, args.train_ratio)
    
    # Create data.yaml
    create_data_yaml(args.target, classes)
    
    print("Dataset split completed!")

if __name__ == "__main__":
    main()
