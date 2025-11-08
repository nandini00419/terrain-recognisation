"""
Deep Learning Terrain Recognition Model Training Script
Trains a CNN model to classify terrain types from images.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Configuration
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
DATA_DIR = '_dataset'
MODEL_SAVE_PATH = 'terrain_model.h5'

# Terrain classes (will be auto-detected from dataset, but you can specify expected classes)
# TERRAIN_CLASSES = ['desert', 'forest', 'mountain', 'plain', 'urban', 'water']

def create_model(num_classes, img_size):
    """
    Create a CNN model using MobileNetV2 as base (transfer learning).
    MobileNetV2 is lightweight and efficient for mobile/web deployment.
    """
    # Load pre-trained MobileNetV2 model
    base_model = MobileNetV2(
        input_shape=(img_size[0], img_size[1], 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model layers initially
    base_model.trainable = False
    
    # Add custom classification head
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def get_data_generators(data_dir, img_size, batch_size):
    """
    Create data generators with augmentation for training and validation.
    Only includes directories that contain image files.
    """
    # Check which directories have images
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')
    valid_classes = []
    class_counts = {}
    
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path):
            files = [f for f in os.listdir(item_path) 
                    if os.path.isfile(os.path.join(item_path, f)) 
                    and f.lower().endswith(image_extensions)]
            if files:
                valid_classes.append(item)
                class_counts[item] = len(files)
    
    if not valid_classes:
        raise ValueError("No directories with images found in dataset!")
    
    if len(valid_classes) < 2:
        raise ValueError(f"Need at least 2 classes with images for training! Found only: {valid_classes}")
    
    print(f"Found {len(valid_classes)} classes with images:")
    for cls in sorted(valid_classes):
        print(f"  {cls}: {class_counts[cls]} images")
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.2,
        fill_mode='nearest',
        validation_split=0.2  # 80% train, 20% validation
    )
    
    # Only rescaling for validation (no augmentation)
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    # Training generator - filter by classes parameter
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        classes=valid_classes,  # Only use directories with images
        seed=42  # Ensure consistent train/val split
    )
    
    # Validation generator
    val_generator = val_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        classes=valid_classes,  # Only use directories with images
        seed=42  # Ensure consistent train/val split
    )
    
    # Verify generators have samples
    if train_generator.samples == 0:
        raise ValueError(f"Training generator has 0 samples! Check your dataset structure and validation_split.")
    
    if val_generator.samples == 0:
        raise ValueError(f"Validation generator has 0 samples! Dataset might be too small or validation_split is incorrect.")
    
    return train_generator, val_generator

def train_model():
    """
    Main training function.
    """
    print("=" * 60)
    print("Deep Learning Terrain Recognition - Model Training")
    print("=" * 60)
    
    # Check if dataset directory exists
    if not os.path.exists(DATA_DIR):
        print(f"Error: Dataset directory '{DATA_DIR}' not found!")
        print("Please organize your images in subdirectories by terrain type:")
        print(f"  {DATA_DIR}/")
        print("    ├── desert/")
        print("    ├── forest/")
        print("    ├── mountain/")
        print("    ├── plain/")
        print("    ├── urban/")
        print("    └── water/")
        return
    
    # Check if dataset is organized into subdirectories
    subdirs = [d for d in os.listdir(DATA_DIR) 
               if os.path.isdir(os.path.join(DATA_DIR, d))]
    files_in_root = [f for f in os.listdir(DATA_DIR) 
                     if os.path.isfile(os.path.join(DATA_DIR, f))]
    
    if not subdirs and files_in_root:
        print(f"\n⚠️  ERROR: Images found in '{DATA_DIR}' but no terrain class subdirectories!")
        print(f"   Found {len(files_in_root)} files in root directory.")
        print("\n   Please organize images into terrain class folders:")
        print("   1. Run: python quick_organize_dataset.py")
        print("   2. Or manually create folders and move images:")
        print(f"      {DATA_DIR}/desert/")
        print(f"      {DATA_DIR}/forest/")
        print(f"      {DATA_DIR}/mountain/")
        print(f"      {DATA_DIR}/plain/")
        print(f"      {DATA_DIR}/urban/")
        print(f"      {DATA_DIR}/water/")
        return
    
    # Get data generators
    print("\n[1/4] Loading and preprocessing data...")
    try:
        train_gen, val_gen = get_data_generators(DATA_DIR, IMG_SIZE, BATCH_SIZE)
    except Exception as e:
        print(f"\n⚠️  Error creating data generators: {e}")
        print("\nThis usually means:")
        print("  1. No images found in terrain class subdirectories")
        print("  2. Images are in wrong format or corrupted")
        print("  3. Dataset structure is incorrect")
        print("\nPlease check your dataset organization and try again.")
        return
    
    # Validate generators
    if train_gen.samples == 0:
        print("\n⚠️  ERROR: No training samples found!")
        print("Please check that images are organized in terrain class subdirectories.")
        return
    
    if val_gen.samples == 0:
        print("\n⚠️  WARNING: No validation samples found!")
        print("Dataset might be too small. Consider using a smaller validation_split.")
    
    num_classes = len(train_gen.class_indices)
    class_names = list(train_gen.class_indices.keys())
    
    print(f"Found {num_classes} terrain classes: {class_names}")
    print(f"Training samples: {train_gen.samples}")
    print(f"Validation samples: {val_gen.samples}")
    
    if train_gen.samples < num_classes:
        print(f"\n⚠️  WARNING: Very few samples per class!")
        print("   Training may not be effective. Consider adding more images.")
    
    # Save class indices mapping for later use
    import json
    with open('class_indices.json', 'w') as f:
        json.dump(train_gen.class_indices, f, indent=2)
    print("Saved class indices to 'class_indices.json'")
    
    # Create model
    print("\n[2/4] Creating model architecture...")
    model = create_model(num_classes, IMG_SIZE)
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nModel Summary:")
    model.summary()
    
    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            MODEL_SAVE_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=0.00001,
            verbose=1
        )
    ]
    
    # Train model
    print("\n[3/4] Training model...")
    print(f"Training for {EPOCHS} epochs...")
    
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    print("\n[4/4] Saving model...")
    model.save(MODEL_SAVE_PATH)
    print(f"Model saved to '{MODEL_SAVE_PATH}'")
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate model
    print("\nEvaluating model on validation set...")
    val_loss, val_accuracy = model.evaluate(val_gen, verbose=1)
    print(f"\nFinal Validation Accuracy: {val_accuracy:.4f}")
    print(f"Final Validation Loss: {val_loss:.4f}")
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)

def plot_training_history(history):
    """
    Plot training and validation accuracy/loss curves.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    print("Training history saved to 'training_history.png'")

if __name__ == '__main__':
    train_model()

