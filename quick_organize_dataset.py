"""
Quick Dataset Organization Script
Automatically organizes images from flat _dataset into terrain class folders.
Distributes images evenly across terrain classes if no pattern is detected.
"""

import os
import shutil
import random
from pathlib import Path

DATA_DIR = '_dataset'
TERRAIN_CLASSES = ['desert', 'forest', 'mountain', 'plain', 'urban', 'water']

def organize_dataset_evenly():
    """
    Organize images evenly across terrain classes.
    This is a temporary solution if images don't have terrain labels in filenames.
    """
    print("=" * 60)
    print("Organizing Dataset - Even Distribution")
    print("=" * 60)
    
    if not os.path.exists(DATA_DIR):
        print(f"Error: {DATA_DIR} directory not found!")
        return
    
    # Create terrain class directories
    print("\nCreating terrain class directories...")
    for cls in TERRAIN_CLASSES:
        dir_path = os.path.join(DATA_DIR, cls)
        os.makedirs(dir_path, exist_ok=True)
        print(f"  Created: {dir_path}")
    
    # Get all image files
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')
    files = [f for f in os.listdir(DATA_DIR) 
             if os.path.isfile(os.path.join(DATA_DIR, f)) 
             and f.lower().endswith(image_extensions)]
    
    if not files:
        print("\nNo image files found in _dataset folder!")
        return
    
    print(f"\nFound {len(files)} image files")
    
    # Distribute images evenly across classes
    random.seed(42)  # For reproducibility
    random.shuffle(files)
    
    num_classes = len(TERRAIN_CLASSES)
    images_per_class = len(files) // num_classes
    remainder = len(files) % num_classes
    
    print(f"\nDistributing images:")
    print(f"  Images per class: ~{images_per_class}")
    
    idx = 0
    for i, cls in enumerate(TERRAIN_CLASSES):
        target_dir = os.path.join(DATA_DIR, cls)
        # Add one extra image to first 'remainder' classes
        count = images_per_class + (1 if i < remainder else 0)
        
        moved = 0
        for j in range(count):
            if idx < len(files):
                src = os.path.join(DATA_DIR, files[idx])
                dst = os.path.join(target_dir, files[idx])
                shutil.move(src, dst)
                moved += 1
                idx += 1
        
        print(f"  {cls}: {moved} images")
    
    print(f"\n[SUCCESS] Successfully organized {len(files)} images into {num_classes} terrain classes!")
    print("\n[NOTE] Images were distributed randomly/evenly.")
    print("   If your images have terrain labels, please:")
    print("   1. Manually organize them into correct terrain class folders")
    print("   2. Or update this script to use filename patterns for automatic organization")

def organize_by_suffix():
    """
    Organize images based on suffix patterns in filenames.
    Adjust the mapping based on your dataset's naming convention.
    """
    print("=" * 60)
    print("Organizing Dataset - By Filename Suffix")
    print("=" * 60)
    
    # Create terrain class directories
    for cls in TERRAIN_CLASSES:
        os.makedirs(os.path.join(DATA_DIR, cls), exist_ok=True)
    
    # Mapping of suffixes to terrain classes
    # UPDATE THIS BASED ON YOUR DATASET!
    suffix_to_class = {
        'h': 'mountain',    # h = hills/mountain
        'i2': 'forest',     # i2 = forest
        't': 'plain',       # t = terrain/plain
        # Add more mappings as needed
        # 'd': 'desert',
        # 'u': 'urban',
        # 'w': 'water',
    }
    
    files = [f for f in os.listdir(DATA_DIR) 
             if os.path.isfile(os.path.join(DATA_DIR, f)) 
             and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    organized = {}
    unorganized = []
    
    for filename in files:
        name = os.path.splitext(filename)[0]
        parts = name.split('_')
        
        if len(parts) > 1:
            suffix = '_'.join(parts[1:]).lower()
            if suffix in suffix_to_class:
                target_class = suffix_to_class[suffix]
                target_dir = os.path.join(DATA_DIR, target_class)
                shutil.move(
                    os.path.join(DATA_DIR, filename),
                    os.path.join(target_dir, filename)
                )
                organized[target_class] = organized.get(target_class, 0) + 1
            else:
                unorganized.append(filename)
        else:
            unorganized.append(filename)
    
    print(f"\nOrganized by suffix:")
    for cls, count in organized.items():
        print(f"  {cls}: {count} images")
    
    if unorganized:
        print(f"\n[WARNING] {len(unorganized)} images couldn't be organized by suffix pattern.")
        print("   These will remain in the root _dataset folder.")
        print("\n   Sample unorganized files:")
        for f in unorganized[:5]:
            print(f"     {f}")
    
    return len(organized), len(unorganized)

def main():
    """Main function."""
    import sys
    
    print("=" * 60)
    print("Quick Dataset Organization")
    print("=" * 60)
    print("\nThis script will organize images in _dataset into terrain class folders.")
    print("\nOptions:")
    print("1. Organize by suffix pattern (if filenames contain terrain info)")
    print("2. Organize evenly across classes (random distribution)")
    print("\n[NOTE] For best results, manually organize images with correct terrain labels!")
    print()
    
    # Check current structure
    if os.path.exists(DATA_DIR):
        subdirs = [d for d in os.listdir(DATA_DIR) 
                  if os.path.isdir(os.path.join(DATA_DIR, d))]
        files = [f for f in os.listdir(DATA_DIR) 
                if os.path.isfile(os.path.join(DATA_DIR, f))]
        
        if subdirs:
            print(f"[WARNING] Found {len(subdirs)} existing subdirectories in _dataset")
            print("   Existing folders:", ', '.join(subdirs[:5]))
        
        if files:
            print(f"[INFO] Found {len(files)} files in _dataset root")
            
            # Check for suffix pattern
            sample_files = files[:10]
            suffixes = set()
            for f in sample_files:
                name = os.path.splitext(f)[0]
                parts = name.split('_')
                if len(parts) > 1:
                    suffixes.add('_'.join(parts[1:]))
            
            if suffixes:
                print(f"   Detected suffixes: {', '.join(list(suffixes)[:5])}")
                print("\n   Attempting to organize by suffix pattern...")
                organize_by_suffix()
            else:
                print("\n   No clear suffix pattern detected.")
                print("   Organizing evenly across classes...")
                organize_dataset_evenly()
        else:
            print("No files found in _dataset folder!")
    else:
        print(f"Error: {DATA_DIR} directory not found!")
    
    print("\n" + "=" * 60)
    print("Next step: Run 'python train_model.py' to train the model")
    print("=" * 60)

if __name__ == '__main__':
    main()

