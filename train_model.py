"""
Hot Dog or Not Hot Dog - Model Training Script
A tribute to Jian Yang's SeeFood app from HBO's Silicon Valley

This script trains a binary image classifier using transfer learning
with MobileNetV2 to detect whether an image contains a hot dog.
"""

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.001

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
TEST_DIR = os.path.join(DATASET_DIR, "test")
MODEL_PATH = os.path.join(BASE_DIR, "hotdog_model.keras")


def create_data_generators():
    """Create training and validation data generators with augmentation."""
    
    # Training data generator with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./127.5,  # Rescale to [-1, 1] for MobileNetV2
        preprocessing_function=lambda x: x - 1,  # Shift to [-1, 1]
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2  # Use 20% for validation
    )
    
    # Test data generator (no augmentation, just rescaling)
    test_datagen = ImageDataGenerator(
        rescale=1./127.5,
        preprocessing_function=lambda x: x - 1
    )
    
    # Create generators
    print("Loading training data...")
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training',
        shuffle=True
    )
    
    print("Loading validation data...")
    validation_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation',
        shuffle=False
    )
    
    print("Loading test data...")
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )
    
    return train_generator, validation_generator, test_generator


def build_model():
    """Build the model using MobileNetV2 with transfer learning."""
    
    print("\nBuilding model with MobileNetV2 backbone...")
    
    # Load MobileNetV2 pretrained on ImageNet
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # Freeze the base model
    base_model.trainable = False
    
    # Build the model
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')  # Binary classification
    ])
    
    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print(model.summary())
    return model


def fine_tune_model(model):
    """Unfreeze some layers for fine-tuning."""
    
    print("\nFine-tuning: Unfreezing top layers of MobileNetV2...")
    
    # Get the base model
    base_model = model.layers[0]
    base_model.trainable = True
    
    # Freeze all layers except the last 30
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    
    # Recompile with a lower learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE / 10),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def train_model():
    """Main training function."""
    
    print("=" * 60)
    print("🌭 HOT DOG OR NOT HOT DOG - Model Training")
    print("   A tribute to Jian Yang's SeeFood from Silicon Valley")
    print("=" * 60)
    
    # Create data generators
    train_gen, val_gen, test_gen = create_data_generators()
    
    # Print class mapping
    print(f"\nClass mapping: {train_gen.class_indices}")
    print(f"  0 = hotdog, 1 = nothotdog")
    
    # Build the model
    model = build_model()
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Phase 1: Train with frozen base
    print("\n" + "=" * 60)
    print("Phase 1: Training with frozen MobileNetV2 base...")
    print("=" * 60)
    
    history1 = model.fit(
        train_gen,
        epochs=EPOCHS // 2,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    # Phase 2: Fine-tune
    print("\n" + "=" * 60)
    print("Phase 2: Fine-tuning top layers...")
    print("=" * 60)
    
    model = fine_tune_model(model)
    
    history2 = model.fit(
        train_gen,
        epochs=EPOCHS,
        initial_epoch=len(history1.history['loss']),
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate on test set
    print("\n" + "=" * 60)
    print("Evaluating on test set...")
    print("=" * 60)
    
    test_loss, test_accuracy = model.evaluate(test_gen, verbose=1)
    print(f"\n🎯 Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"📉 Test Loss: {test_loss:.4f}")
    
    # Save the final model
    model.save(MODEL_PATH)
    print(f"\n✅ Model saved to: {MODEL_PATH}")
    
    # Print predictions interpretation
    print("\n" + "=" * 60)
    print("📝 How to interpret predictions:")
    print("   - Output < 0.5 → HOT DOG 🌭")
    print("   - Output >= 0.5 → NOT HOT DOG ❌")
    print("=" * 60)
    
    return model, test_accuracy


if __name__ == "__main__":
    # Check TensorFlow version
    print(f"TensorFlow version: {tf.__version__}")
    
    # Check GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU(s) available: {len(gpus)}")
        for gpu in gpus:
            print(f"  - {gpu}")
    else:
        print("No GPU available, training on CPU (this may be slow)")
    
    # Train the model
    model, accuracy = train_model()
    
    if accuracy >= 0.9:
        print("\n🎉 Excellent! Model achieved >90% accuracy!")
    elif accuracy >= 0.8:
        print("\n👍 Good! Model achieved >80% accuracy.")
    else:
        print("\n⚠️ Model accuracy is below 80%. Consider more training data or tuning.")
