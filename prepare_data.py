"""
Data preparation script for pothole detection model.
Organizes images into train/validation/test splits.
"""

import os
import shutil
from pathlib import Path
import random
from tqdm import tqdm

def prepare_dataset(source_dir="Pothole_img", output_dir="dataset", 
                   train_split=0.7, val_split=0.15, test_split=0.15, seed=42):
    """
    Organize images into train/validation/test splits.
    
    Args:
        source_dir: Directory containing pothole images
        output_dir: Output directory for organized dataset
        train_split: Fraction for training set
        val_split: Fraction for validation set
        test_split: Fraction for test set
        seed: Random seed for reproducibility
    """
    # Validate splits
    assert abs(train_split + val_split + test_split - 1.0) < 0.001, \
        "Splits must sum to 1.0"
    
    random.seed(seed)
    
    # Create directory structure
    splits = ['train', 'val', 'test']
    classes = ['pothole', 'no_pothole']
    
    for split in splits:
        for class_name in classes:
            Path(f"{output_dir}/{split}/{class_name}").mkdir(parents=True, exist_ok=True)
    
    # Get all pothole images
    source_path = Path(source_dir)
    pothole_images = list(source_path.glob("*.jpg")) + list(source_path.glob("*.jpeg")) + \
                     list(source_path.glob("*.png"))
    
    print(f"Found {len(pothole_images)} pothole images")
    
    # Shuffle and split pothole images
    random.shuffle(pothole_images)
    total = len(pothole_images)
    train_end = int(total * train_split)
    val_end = train_end + int(total * val_split)
    
    train_images = pothole_images[:train_end]
    val_images = pothole_images[train_end:val_end]
    test_images = pothole_images[val_end:]
    
    # Copy pothole images to respective folders
    print("Copying pothole images...")
    for img_path in tqdm(train_images, desc="Train"):
        shutil.copy2(img_path, f"{output_dir}/train/pothole/{img_path.name}")
    
    for img_path in tqdm(val_images, desc="Validation"):
        shutil.copy2(img_path, f"{output_dir}/val/pothole/{img_path.name}")
    
    for img_path in tqdm(test_images, desc="Test"):
        shutil.copy2(img_path, f"{output_dir}/test/pothole/{img_path.name}")
    
    print(f"\nDataset organized:")
    print(f"  Train: {len(train_images)} pothole images")
    print(f"  Validation: {len(val_images)} pothole images")
    print(f"  Test: {len(test_images)} pothole images")
    print(f"\nNOTE: You need to add non-pothole images to:")
    print(f"  - {output_dir}/train/no_pothole/")
    print(f"  - {output_dir}/val/no_pothole/")
    print(f"  - {output_dir}/test/no_pothole/")
    print(f"\nFor balanced training, try to match the number of images in each class.")

if __name__ == "__main__":
    prepare_dataset()
