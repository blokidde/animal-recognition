import argparse
import json
import os
import pandas as pd
from pathlib import Path
from typing import Optional

DEFAULT_CLASS_INDICES = Path("train/class_indices.json")

def load_class_ids(class_indices_path: str) -> dict[str, int]:
    path = Path(class_indices_path)
    if not path.exists():
        raise FileNotFoundError(f"Class mapping not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def get_class_id(species: str, class_ids: dict[str, int]) -> Optional[int]:
    """Get class ID for species name."""
    return class_ids.get(species)

def find_image_in_dataset(image_name: str, dataset_dir: str) -> Optional[str]:
    """Find which split (train/val) contains the image."""
    train_dir = os.path.join(dataset_dir, 'images', 'train')
    val_dir = os.path.join(dataset_dir, 'images', 'val')
    
    if os.path.exists(os.path.join(train_dir, image_name)):
        return 'train'
    elif os.path.exists(os.path.join(val_dir, image_name)):
        return 'val'
    
    return None

def write_yolo_label(label_path: str, detections: list, dry_run: bool = False) -> None:
    """Write YOLO format label file."""
    if dry_run:
        print(f"[DRY RUN] Would write to {label_path}:")
        for detection in detections:
            print(f"  {detection}")
        return
    
    os.makedirs(os.path.dirname(label_path), exist_ok=True)
    
    with open(label_path, 'w') as f:
        for detection in detections:
            f.write(f"{detection}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', required=True, help='Path to results CSV file')
    parser.add_argument('--dataset_dir', default='datasets/animals')
    parser.add_argument('--class_indices', default=str(DEFAULT_CLASS_INDICES))
    parser.add_argument('--min_yolo_conf', type=float, default=0.25)
    parser.add_argument('--min_species_conf', type=float, default=0.6)
    parser.add_argument('--dry_run', action='store_true', default=False)
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results):
        raise FileNotFoundError(f"Results CSV not found: {args.results}")
    
    # Load results
    df = pd.read_csv(args.results)
    class_ids = load_class_ids(args.class_indices)
    
    # Filter by confidence thresholds
    filtered_df = df[
        (df['yolo_conf'] >= args.min_yolo_conf) & 
        (df['species_conf'] >= args.min_species_conf) &
        (df['species'] != 'unknown')
    ]
    
    print(f"Loaded {len(df)} detections, {len(filtered_df)} passed confidence filters")
    
    # Dataset directory (assumed YOLO structure)
    dataset_dir = args.dataset_dir
    
    if not os.path.exists(dataset_dir):
        print(f"Warning: Dataset directory not found: {dataset_dir}")
        print("Labels will be written relative to current directory")
        dataset_dir = "."
    
    # Group by image
    processed_images = 0
    written_labels = 0
    
    for image_path, group in filtered_df.groupby('image_path'):
        image_name = os.path.basename(image_path)
        
        # Find which split this image belongs to
        split = find_image_in_dataset(image_name, dataset_dir)
        
        if split is None:
            print(f"Warning: Image {image_name} not found in dataset, skipping")
            continue
        
        # Prepare YOLO format labels
        yolo_labels = []
        
        for _, row in group.iterrows():
            class_id = get_class_id(row['species'], class_ids)
            if class_id is None:
                print(f"Warning: Unknown species {row['species']}, skipping")
                continue
            
            label_line = f"{class_id} {row['x_center_n']:.6f} {row['y_center_n']:.6f} {row['w_n']:.6f} {row['h_n']:.6f}"
            yolo_labels.append(label_line)
        
        if yolo_labels:
            # Write label file
            label_filename = os.path.splitext(image_name)[0] + '.txt'
            label_path = os.path.join(dataset_dir, 'labels', split, label_filename)
            
            write_yolo_label(label_path, yolo_labels, args.dry_run)
            
            processed_images += 1
            written_labels += len(yolo_labels)
    
    print(f"Processed {processed_images} images, wrote {written_labels} labels")
    
    if args.dry_run:
        print("Dry run completed - no files were actually written")

if __name__ == "__main__":
    main()
