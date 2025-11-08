"""
Dataset Organization Script
Organizes images from flat _dataset folder into terrain class subdirectories.
If images have labels in filenames or need manual organization, this script helps.
"""

import os
import shutil
from pathlib import Path
import re

DATA_DIR = '_dataset'
TERRAIN_CLASSES = ['desert', 'forest', 'mountain', 'plain', 'urban', 'water']

def analyze_dataset():
    """Analyze the current dataset structure."""
    print("=" * 60)
    print("Dataset Analysis")
    print("=" * 60)
    
    if not os.path.exists(DATA_DIR):
        print(f"Error: {DATA_DIR} directory not found!")
        return
    
    # Check for subdirectories
    subdirs = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    files = [f for f in os.listdir(DATA_DIR) if os.path.isfile(os.path.join(DATA_DIR, f))]
    
    print(f"\nSubdirectories found: {len(subdirs)}")
    if subdirs:
        print(f"  {', '.join(subdirs[:10])}")
    
    print(f"\nFiles found: {len(files)}")
    
    if files:
        print("\nSample filenames:")
        for f in files[:10]:
            print(f"  {f}")
        
        # Analyze filename patterns
        print("\nFilename patterns:")
        suffixes = set()
        prefixes = set()
        for f in files[:100]:  # Sample first 100
            name = os.path.splitext(f)[0]
            parts = name.split('_')
            if len(parts) > 1:
                suffixes.add('_'.join(parts[1:]))
            if parts:
                prefixes.add(parts[0])
        
        print(f"  Unique prefixes (first 10): {list(prefixes)[:10]}")
        print(f"  Unique suffixes (first 10): {list(suffixes)[:10]}")
    
    return len(subdirs), len(files)

def create_class_directories():
    """Create terrain class directories if they don't exist."""
    for cls in TERRAIN_CLASSES:
        dir_path = os.path.join(DATA_DIR, cls)
        os.makedirs(dir_path, exist_ok=True)
    print(f"\nCreated {len(TERRAIN_CLASSES)} terrain class directories")

def organize_by_suffix_mapping():
    """
    Organize images by suffix mapping.
    This assumes filenames have patterns like: ID_suffix.png
    You may need to adjust the mapping based on your dataset.
    """
    print("\n" + "=" * 60)
    print("Organizing Images by Suffix Pattern")
    print("=" * 60)
    
    # Map suffixes to terrain classes
    # Adjust this mapping based on your actual dataset
    suffix_mapping = {
        'h': 'mountain',   # Example: h = mountain (hills)
        'i2': 'forest',    # Example: i2 = forest
        't': 'plain',      # Example: t = plain/terrain
        # Add more mappings based on your dataset
    }
    
    files = [f for f in os.listdir(DATA_DIR) 
             if os.path.isfile(os.path.join(DATA_DIR, f)) 
             and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    organized = 0
    unorganized = []
    
    for filename in files:
        name = os.path.splitext(filename)[0]
        parts = name.split('_')
        
        if len(parts) > 1:
            suffix = '_'.join(parts[1:])
            if suffix in suffix_mapping:
                target_class = suffix_mapping[suffix]
                target_dir = os.path.join(DATA_DIR, target_class)
                shutil.move(
                    os.path.join(DATA_DIR, filename),
                    os.path.join(target_dir, filename)
                )
                organized += 1
            else:
                unorganized.append(filename)
        else:
            unorganized.append(filename)
    
    print(f"\nOrganized {organized} images into terrain classes")
    print(f"Unorganized images: {len(unorganized)}")
    
    if unorganized:
        print("\nUnorganized images (first 10):")
        for f in unorganized[:10]:
            print(f"  {f}")
        print("\nYou may need to manually organize these or update the suffix mapping.")
    
    return organized, len(unorganized)

def split_existing_dataset():
    """
    Split existing organized dataset if images are already in subdirectories
    but not in the expected terrain class folders.
    """
    subdirs = [d for d in os.listdir(DATA_DIR) 
               if os.path.isdir(os.path.join(DATA_DIR, d)) 
               and d not in TERRAIN_CLASSES]
    
    if not subdirs:
        return 0
    
    print(f"\nFound {len(subdirs)} existing subdirectories")
    print("If these contain terrain-classified images, map them to terrain classes")
    
    # Example: map existing folders to terrain classes
    # Adjust based on your actual folder names
    folder_mapping = {}
    
    for subdir in subdirs:
        print(f"\n  '{subdir}' -> ?")
        # You would manually map these or use a naming convention
    
    return 0

def organize_manually_interactive():
    """
    Interactive script to help organize images manually.
    Shows images and lets user assign terrain classes.
    """
    print("\n" + "=" * 60)
    print("Interactive Image Organization")
    print("=" * 60)
    print("\nThis would open an interactive tool to organize images.")
    print("For now, please organize images manually into:")
    for cls in TERRAIN_CLASSES:
        print(f"  {DATA_DIR}/{cls}/")

def main():
    """Main function."""
    print("=" * 60)
    print("Dataset Organization Tool")
    print("=" * 60)
    
    # Analyze current structure
    num_subdirs, num_files = analyze_dataset()
    
    if num_subdirs == 0 and num_files > 0:
        print("\n" + "=" * 60)
        print("Dataset has images but no terrain class folders!")
        print("=" * 60)
        print("\nOptions:")
        print("1. Organize by filename pattern (if filenames contain terrain info)")
        print("2. Manual organization (recommended for best results)")
        print("\nCreating terrain class directories...")
        create_class_directories()
        
        print("\nIf your filenames contain terrain information:")
        print("  - Edit organize_dataset.py")
        print("  - Update the suffix_mapping dictionary")
        print("  - Run: organize_by_suffix_mapping()")
        print("\nOtherwise, manually organize images into:")
        for cls in TERRAIN_CLASSES:
            print(f"  {DATA_DIR}/{cls}/")
        
        # Try to organize by common patterns
        print("\nAttempting to organize by filename patterns...")
        organized, unorganized = organize_by_suffix_mapping()
        
        if unorganized > 0:
            print(f"\n⚠️  {unorganized} images could not be automatically organized.")
            print("Please manually organize them into terrain class folders.")
    
    elif num_subdirs > 0:
        # Check if terrain class folders exist
        existing_classes = [d for d in os.listdir(DATA_DIR) 
                          if os.path.isdir(os.path.join(DATA_DIR, d)) 
                          and d in TERRAIN_CLASSES]
        
        if len(existing_classes) == len(TERRAIN_CLASSES):
            print("\n✅ Dataset is already organized!")
            print(f"Found terrain classes: {', '.join(existing_classes)}")
        else:
            print(f"\nFound {num_subdirs} subdirectories, but may not match terrain classes.")
            print("Expected terrain classes:", ', '.join(TERRAIN_CLASSES))
    
    print("\n" + "=" * 60)

if __name__ == '__main__':
    main()

