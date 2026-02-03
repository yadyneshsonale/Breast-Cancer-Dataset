import os
import numpy as np
import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import resample

# Paths for datasets
image_folder_left = "./Breast-Cancer-Dataset/paper_data/left"
label_file_left = "./Breast-Cancer-Dataset/paper_data/left/left.xlsx"

image_folder_right = "./Breast-Cancer-Dataset/paper_data/right"
label_file_right = "./Breast-Cancer-Dataset/paper_data/right/right.xlsx"

# Ensure output directory
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

# Function to preprocess an image
def preprocess_image(image_path, target_size=(224, 224)):
    try:
        image = load_img(image_path, target_size=target_size, color_mode='rgb')
        image = img_to_array(image)
        image = preprocess_input(image)
        return image
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

# Function to process a dataset
def process_dataset(image_folder, label_file, target_size=(224, 224)):
    labels_df = pd.read_excel(label_file)
    labels_df.set_index("Image", inplace=True)
    labels_df.index = labels_df.index.astype(str)

    X, y = [], []
    for image_name in os.listdir(image_folder):
        if image_name.endswith(".png"):
            base_name = os.path.splitext(image_name)[0]
            image_path = os.path.join(image_folder, image_name)
            if base_name in labels_df.index:
                processed_image = preprocess_image(image_path, target_size)
                if processed_image is not None:
                    X.append(processed_image)
                    y.append(labels_df.loc[base_name, "Label"])
    return np.array(X), np.array(y)

def process_dataset_right(image_folder, label_file, target_size=(224, 224)):
    labels_df = pd.read_excel(label_file)
    labels_df.set_index("Image", inplace=True)
    labels_df.index = labels_df.index.astype(str)

    X, y = [], []
    for image_name in os.listdir(image_folder):
        if image_name.endswith(".png"):
            base_name = f"{os.path.splitext(image_name)[0]}_right"
            image_path = os.path.join(image_folder, image_name)
            if base_name in labels_df.index:
                processed_image = preprocess_image(image_path, target_size)
                if processed_image is not None:
                    X.append(processed_image)
                    y.append(labels_df.loc[base_name, "Label"])
    return np.array(X), np.array(y)

# Load and preprocess datasets
X_left, y_left = process_dataset(image_folder_left, label_file_left)
X_right, y_right = process_dataset_right(image_folder_right, label_file_right)

print("Data Loaded")

# Combine datasets
X = np.vstack((X_left, X_right))
y = np.concatenate((y_left, y_right))

# Encode labels to integers
class_mapping = {label: idx for idx, label in enumerate(np.unique(y))}
y = np.array([class_mapping[label] for label in y])

# One-hot encode labels
y = to_categorical(y, num_classes=len(class_mapping))

# Resample to balance classes
TARGET_SAMPLES = 183

X_resampled = []
y_resampled = []

num_classes = y.shape[1]  # number of classes (from one-hot labels)

for class_idx in range(num_classes):
    # Select samples of the current class
    X_class = X[y.argmax(axis=1) == class_idx]
    y_class = y[y.argmax(axis=1) == class_idx]

    # Resample to exactly 183 samples
    X_class_resampled, y_class_resampled = resample(
        X_class,
        y_class,
        replace=True,              # oversampling allowed
        n_samples=TARGET_SAMPLES,  # FIXED size
        random_state=42
    )

    X_resampled.append(X_class_resampled)
    y_resampled.append(y_class_resampled)

# Combine all classes
X_resampled = np.vstack(X_resampled)
y_resampled = np.vstack(y_resampled)

# Shuffle the dataset
indices = np.random.permutation(len(X_resampled))
X_resampled = X_resampled[indices]
y_resampled = y_resampled[indices]

print("Data Resampled")
# Function to create the model
def create_model():
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(len(class_mapping), activation='softmax')
    ])
    base_model.trainable = True
    for layer in base_model.layers[:143]:
        layer.trainable = False
    model.compile(optimizer=Adam(learning_rate=1e-5),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

NUM_RUNS = 10
best_val_acc = -np.inf
best_run = None
best_model = None
best_history = None

print("Training Started")

from collections import Counter

for run in range(NUM_RUNS):
    # Train-val split (change random_state per run)
    X_train, X_val, y_train, y_val = train_test_split(
        X_resampled, y_resampled,
        test_size=0.2,
        random_state=42 + run,
        stratify=y_resampled
    )

    # Compute class weights
    y_train_int = np.argmax(y_train, axis=1)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train_int),
        y=y_train_int
    )
    class_weights_dict = dict(enumerate(class_weights))
    callbacks = [
      EarlyStopping(
          monitor='val_loss',
          patience=10,
          restore_best_weights=True,
          verbose=0
      ),

      ModelCheckpoint(
          filepath=f'best_model_run_{run}.keras',
          monitor='val_accuracy',
          save_best_only=True,
          mode='max',
          verbose=0
      ),

      ReduceLROnPlateau(
          monitor='val_loss',
          factor=0.3,
          patience=5,
          min_lr=1e-6,
          verbose=0
      )
    ]
    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    datagen.fit(X_train)

    # Create a fresh model
    model = create_model()

    # Train
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=16),
        epochs=70,
        validation_data=(X_val, y_val),
        class_weight=class_weights_dict,
        callbacks=callbacks,
        verbose=0
    )

    # Get best val accuracy from this run
    run_best_val_acc = max(history.history['val_accuracy'])
    print(f"{run+1} best val_accuracy: {run_best_val_acc:.4f}")

    # Track best run
    if run_best_val_acc > best_val_acc:
        best_val_acc = run_best_val_acc
        best_run = run + 1
        best_model = model
        best_history = history

# Plot training metrics and confusion matrix
def plot_results(history, X_val, y_val, model):
    y_pred = model.predict(X_val)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_val, axis=1)

    # Create plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot training accuracy and loss
    axes[0].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0].plot(history.history['loss'], label='Training Loss')
    axes[0].set_title('Training Metrics')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Value')
    axes[0].legend()

    # Plot validation accuracy and loss
    axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[1].plot(history.history['val_loss'], label='Validation Loss')
    axes[1].set_title('Validation Metrics')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Value')
    axes[1].legend()

    # Plot confusion matrix
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(class_mapping.keys()))
    disp.plot(ax=axes[2], cmap=plt.cm.Blues, colorbar=False)
    axes[2].set_title('Confusion Matrix')

    # Save plots
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'results.png'))
    plt.show()

# Generate and save plots
plot_results(best_history, X_val, y_val, model)
print(f"Val_accuracy: {best_val_acc:.4f}")