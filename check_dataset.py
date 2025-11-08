"""
Check dataset structure and count images in each class folder.
"""

import os

DATA_DIR = '_dataset'
image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')

print("=" * 60)
print("Dataset Structure Check")
print("=" * 60)

if not os.path.exists(DATA_DIR):
    print(f"Error: {DATA_DIR} directory not found!")
    exit(1)

# Check subdirectories
subdirs = [d for d in os.listdir(DATA_DIR) 
           if os.path.isdir(os.path.join(DATA_DIR, d))]

files_in_root = [f for f in os.listdir(DATA_DIR) 
                 if os.path.isfile(os.path.join(DATA_DIR, f)) 
                 and f.lower().endswith(image_extensions)]

print(f"\nFiles in root: {len(files_in_root)}")
print(f"Subdirectories: {len(subdirs)}")

if subdirs:
    print("\nImage counts per class:")
    total_images = 0
    for subdir in sorted(subdirs):
        subdir_path = os.path.join(DATA_DIR, subdir)
        images = [f for f in os.listdir(subdir_path) 
                 if os.path.isfile(os.path.join(subdir_path, f)) 
                 and f.lower().endswith(image_extensions)]
        count = len(images)
        total_images += count
        status = "OK" if count > 0 else "EMPTY"
        print(f"  {subdir:15s}: {count:5d} images [{status}]")
    
    print(f"\nTotal images: {total_images}")
    
    # Check which classes have enough images
    classes_with_images = [subdir for subdir in subdirs 
                          if len([f for f in os.listdir(os.path.join(DATA_DIR, subdir)) 
                                 if os.path.isfile(os.path.join(DATA_DIR, subdir, f)) 
                                 and f.lower().endswith(image_extensions)]) > 0]
    
    print(f"\nClasses with images: {len(classes_with_images)}")
    print(f"  {', '.join(sorted(classes_with_images))}")
    
    if len(classes_with_images) < 2:
        print("\n[WARNING] Need at least 2 classes with images for training!")
else:
    print("\n[ERROR] No subdirectories found!")
    if files_in_root:
        print(f"  Found {len(files_in_root)} images in root directory.")
        print("  Please organize images into class subdirectories.")

print("\n" + "=" * 60)

