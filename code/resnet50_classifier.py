"""
ResNet-50 Breast Cancer Classification

This script classifies thermal breast images into three categories:
- PB (Possibly Benign)
- PM (Possibly Malignant)
- N (Normal)

Using pre-trained ResNet-50 with additional layers for classification.
Dataset is resampled to 550 samples (440 training, 110 testing).
"""

import os
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils import resample, shuffle
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configuration
IMG_SIZE = 244
NUM_CHANNELS = 3
NUM_CLASSES = 3
BATCH_SIZE = 32
EPOCHS = 70
TOTAL_SAMPLES = 550
TRAIN_SAMPLES = 440
TEST_SAMPLES = 110

# Dataset paths
BASE_PATH = "../paper_data"
LEFT_PATH = os.path.join(BASE_PATH, "left")
RIGHT_PATH = os.path.join(BASE_PATH, "right")

# Label mapping
LABEL_MAP = {"N": 0, "PB": 1, "PM": 2}
LABEL_NAMES = ["Normal", "Possibly Benign", "Possibly Malignant"]


def load_image(image_path):
    """Load and preprocess a single image."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Resize to 244x244
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return img


def load_dataset():
    """Load all images from left and right folders with their labels."""
    images = []
    labels = []
    
    # Load left breast images
    left_df = pd.read_excel(os.path.join(LEFT_PATH, "left.xlsx"))
    for _, row in left_df.iterrows():
        img_name = row['Image']
        label = row['Label']
        img_path = os.path.join(LEFT_PATH, f"{img_name}.png")
        
        if os.path.exists(img_path):
            img = load_image(img_path)
            if img is not None:
                images.append(img)
                labels.append(LABEL_MAP[label])
    
    # Load right breast images
    right_df = pd.read_excel(os.path.join(RIGHT_PATH, "right.xlsx"))
    for _, row in right_df.iterrows():
        img_name = row['Image']
        label = row['Label']
        img_path = os.path.join(RIGHT_PATH, f"{img_name}.png")
        
        if os.path.exists(img_path):
            img = load_image(img_path)
            if img is not None:
                images.append(img)
                labels.append(LABEL_MAP[label])
    
    return np.array(images), np.array(labels)


def resample_dataset(images, labels, target_size=TOTAL_SAMPLES):
    """
    Resample dataset to handle class imbalance using oversampling.
    Target: 550 total samples with balanced classes.
    """
    # Get samples per class
    samples_per_class = target_size // NUM_CLASSES
    
    resampled_images = []
    resampled_labels = []
    
    for class_idx in range(NUM_CLASSES):
        # Get all samples of this class
        class_mask = labels == class_idx
        class_images = images[class_mask]
        class_labels = labels[class_mask]
        
        # Resample to target size per class
        if len(class_images) < samples_per_class:
            # Oversample if fewer samples
            indices = np.random.choice(len(class_images), samples_per_class, replace=True)
        else:
            # Undersample if more samples
            indices = np.random.choice(len(class_images), samples_per_class, replace=False)
        
        resampled_images.extend(class_images[indices])
        resampled_labels.extend([class_idx] * samples_per_class)
    
    # Handle remainder to get exactly target_size
    remainder = target_size - (samples_per_class * NUM_CLASSES)
    if remainder > 0:
        # Add remaining samples from random classes
        for i in range(remainder):
            class_idx = i % NUM_CLASSES
            class_mask = labels == class_idx
            class_images = images[class_mask]
            idx = np.random.randint(len(class_images))
            resampled_images.append(class_images[idx])
            resampled_labels.append(class_idx)
    
    return np.array(resampled_images), np.array(resampled_labels)


def create_data_augmentation():
    """Create data augmentation for training."""
    return ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.1,
        fill_mode='nearest'
    )


def create_resnet50_model():
    """
    Create ResNet-50 model with additional layers for breast tumor classification.
    Pre-trained on ImageNet.
    """
    # Load pre-trained ResNet-50 without top layers
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, NUM_CHANNELS)
    )
    
    # Freeze base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Add custom layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    output = Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=output)
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def plot_training_history(history):
    """Plot training and validation accuracy/loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy plot
    axes[0].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[0].set_title('Model Accuracy', fontsize=14)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3)
    
    # Loss plot
    axes[1].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[1].set_title('Model Loss', fontsize=14)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../results/training_history.png', dpi=150)
    plt.show()


def plot_roc_curves(y_test, y_pred_prob):
    """Plot ROC curves for multi-class classification."""
    # Binarize the labels for ROC
    y_test_bin = y_test
    
    plt.figure(figsize=(10, 8))
    colors = ['blue', 'green', 'red']
    
    for i, (color, name) in enumerate(zip(colors, LABEL_NAMES)):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2, 
                 label=f'{name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves for Multi-Class Classification', fontsize=14)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.savefig('../results/roc_curves.png', dpi=150)
    plt.show()


def plot_confusion_matrix(y_true, y_pred):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix', fontsize=14)
    plt.colorbar()
    
    tick_marks = np.arange(NUM_CLASSES)
    plt.xticks(tick_marks, LABEL_NAMES, rotation=45, ha='right')
    plt.yticks(tick_marks, LABEL_NAMES)
    
    # Add text annotations
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha='center', va='center',
                     color='white' if cm[i, j] > thresh else 'black')
    
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig('../results/confusion_matrix.png', dpi=150)
    plt.show()


def main():
    # Create results directory
    os.makedirs('../results', exist_ok=True)
    
    print("=" * 60)
    print("ResNet-50 Breast Cancer Classification")
    print("=" * 60)
    
    # Step 1: Load dataset
    print("\n[1/6] Loading dataset...")
    images, labels = load_dataset()
    print(f"    Loaded {len(images)} images")
    print(f"    Original class distribution:")
    for i, name in enumerate(LABEL_NAMES):
        count = np.sum(labels == i)
        print(f"        {name}: {count}")
    
    # Step 2: Resample dataset to handle class imbalance
    print(f"\n[2/6] Resampling dataset to {TOTAL_SAMPLES} samples...")
    images, labels = resample_dataset(images, labels, TOTAL_SAMPLES)
    print(f"    Resampled class distribution:")
    for i, name in enumerate(LABEL_NAMES):
        count = np.sum(labels == i)
        print(f"        {name}: {count}")
    
    # Shuffle the dataset
    images, labels = shuffle(images, labels, random_state=42)
    
    # Normalize images
    images = images.astype('float32') / 255.0
    
    # Step 3: Split dataset (80% train, 20% test)
    print(f"\n[3/6] Splitting dataset (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"    Training samples: {len(X_train)}")
    print(f"    Testing samples: {len(X_test)}")
    
    # One-hot encode labels
    y_train_cat = to_categorical(y_train, NUM_CLASSES)
    y_test_cat = to_categorical(y_test, NUM_CLASSES)
    
    # Step 4: Create model
    print("\n[4/6] Creating ResNet-50 model...")
    model = create_resnet50_model()
    model.summary()
    
    # Step 5: Train model with data augmentation
    print(f"\n[5/6] Training model for {EPOCHS} epochs...")
    
    # Data augmentation
    datagen = create_data_augmentation()
    datagen.fit(X_train)
    
    # Callbacks
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6),
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    ]
    
    # Train
    history = model.fit(
        datagen.flow(X_train, y_train_cat, batch_size=BATCH_SIZE),
        steps_per_epoch=len(X_train) // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_test, y_test_cat),
        callbacks=callbacks,
        verbose=1
    )
    
    # Step 6: Evaluate model
    print("\n[6/6] Evaluating model...")
    
    # Test accuracy
    test_loss, test_accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"\n    Test Loss: {test_loss:.4f}")
    print(f"    Test Accuracy: {test_accuracy * 100:.2f}%")
    
    # Predictions
    y_pred_prob = model.predict(X_test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    # Classification report
    print("\n    Classification Report:")
    print(classification_report(y_test, y_pred, target_names=LABEL_NAMES))
    
    # Plot results
    print("\nGenerating plots...")
    plot_training_history(history)
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curves(y_test_cat, y_pred_prob)
    
    # Save model
    model.save('../results/resnet50_breast_cancer_model.h5')
    print("\nModel saved to '../results/resnet50_breast_cancer_model.h5'")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
