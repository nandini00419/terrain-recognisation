"""
Setup verification script for Terrain Recognition System
Checks if all required files and directories are in place.
"""

import os
import sys

def check_file_exists(filepath, description):
    """Check if a file exists."""
    exists = os.path.exists(filepath)
    status = "[OK]" if exists else "[MISSING]"
    print(f"{status} {description}: {filepath}")
    return exists

def check_directory_exists(dirpath, description):
    """Check if a directory exists."""
    exists = os.path.isdir(dirpath)
    status = "[OK]" if exists else "[MISSING]"
    print(f"{status} {description}: {dirpath}")
    return exists

def main():
    print("=" * 60)
    print("Terrain Recognition System - Setup Verification")
    print("=" * 60)
    print()
    
    all_ok = True
    
    # Check required directories
    print("Checking directories...")
    all_ok &= check_directory_exists("templates", "Templates directory")
    all_ok &= check_directory_exists("static", "Static directory")
    all_ok &= check_directory_exists("static/css", "CSS directory")
    all_ok &= check_directory_exists("static/js", "JavaScript directory")
    all_ok &= check_directory_exists("static/uploads", "Uploads directory")
    all_ok &= check_directory_exists("_dataset", "Dataset directory")
    print()
    
    # Check required files
    print("Checking files...")
    all_ok &= check_file_exists("train_model.py", "Training script")
    all_ok &= check_file_exists("app.py", "Flask application")
    all_ok &= check_file_exists("requirements.txt", "Requirements file")
    all_ok &= check_file_exists("README.md", "README file")
    all_ok &= check_file_exists("templates/index.html", "Index template")
    all_ok &= check_file_exists("templates/result.html", "Result template")
    all_ok &= check_file_exists("static/css/style.css", "Stylesheet")
    all_ok &= check_file_exists("static/js/main.js", "Main JavaScript")
    all_ok &= check_file_exists("static/js/result.js", "Result JavaScript")
    print()
    
    # Check optional files (may not exist yet)
    print("Checking optional files (may not exist yet)...")
    model_exists = check_file_exists("terrain_model.h5", "Trained model")
    class_indices_exists = check_file_exists("class_indices.json", "Class indices")
    print()
    
    # Summary
    print("=" * 60)
    if all_ok:
        print("[SUCCESS] All required files and directories are in place!")
        print()
        if not model_exists:
            print("[NOTE] Model not found. You need to train the model first:")
            print("   python train_model.py")
        else:
            print("[SUCCESS] Model found! You can start the web application:")
            print("   python app.py")
    else:
        print("[ERROR] Some required files or directories are missing.")
        print("   Please check the errors above.")
    print("=" * 60)
    
    return 0 if all_ok else 1

if __name__ == '__main__':
    sys.exit(main())

