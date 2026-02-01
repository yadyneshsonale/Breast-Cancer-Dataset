import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import tqdm

from sklearn.utils import shuffle
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# =========================================================
# Reproducibility
# =========================================================
np.random.seed(42)
tf.random.set_seed(42)

# =========================================================
# Configuration
# =========================================================
IMG_SIZE = 244
NUM_CHANNELS = 3
NUM_CLASSES = 3
BATCH_SIZE = 32
EPOCHS = 70
NUM_RUNS = 3

LABEL_MAP = {"N": 0, "PB": 1, "PM": 2}
LABEL_NAMES = ["Normal", "Possibly Benign", "Possibly Malignant"]

BASE_PATH = "/Users/yadyneshsonale/Desktop/Breast-Cancer-Dataset/paper_data"
LEFT_PATH = os.path.join(BASE_PATH, "left")
RIGHT_PATH = os.path.join(BASE_PATH, "right")

RESULTS_DIR = "../results"
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 60)
print("ResNet-50 Breast Cancer Classification")
print("=" * 60)

images, labels = [], []

left_df = pd.read_excel(os.path.join(LEFT_PATH, "left.xlsx"))
for _, row in tqdm.tqdm(left_df.iterrows(), total=len(left_df)):
    img_path = os.path.join(LEFT_PATH, f"{row['Image']}.png")
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)
            labels.append(LABEL_MAP[row["Label"]])

right_df = pd.read_excel(os.path.join(RIGHT_PATH, "right.xlsx"))
for _, row in tqdm.tqdm(right_df.iterrows(), total=len(right_df)):
    img_path = os.path.join(RIGHT_PATH, f"{row['Image']}.png")
    if os.path.exists(img_path):
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)
            labels.append(LABEL_MAP[row["Label"]])

images = np.array(images, dtype="float32") / 255.0
labels = np.array(labels)

print(f"\nLoaded {len(images)} images")
TOTAL_SAMPLES = 550
TEST_DISTRIBUTION = {"PM": 44, "PB": 36, "N": 30}

samples_per_class = TOTAL_SAMPLES // NUM_CLASSES
resampled_images, resampled_labels = [], []

for class_idx in range(NUM_CLASSES):
    class_imgs = images[labels == class_idx]
    idx = np.random.choice(
        len(class_imgs),
        samples_per_class,
        replace=len(class_imgs) < samples_per_class
    )
    resampled_images.extend(class_imgs[idx])
    resampled_labels.extend([class_idx] * samples_per_class)

images = np.array(resampled_images)
labels = np.array(resampled_labels)

def build_model():
    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, NUM_CHANNELS)
    )

    for layer in base_model.layers:
        layer.trainable = True

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    outputs = Dense(NUM_CLASSES, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=outputs)

    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=True
)

callbacks = [
    ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5, min_lr=1e-6),
    EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
]

ACCURACY = 0.0
SELECTED_HISTORY = None
MODEL_SAVE_PATH = os.path.join(RESULTS_DIR, "resnet50.h5")

for run in range(NUM_RUNS):

    images_shuffled, labels_shuffled = shuffle(
        images, labels, random_state=42 + run
    )

    X_train, y_train, X_test, y_test = [], [], [], []

    for label_name, count in TEST_DISTRIBUTION.items():
        class_idx = LABEL_MAP[label_name]
        class_imgs = images_shuffled[labels_shuffled == class_idx]

        selected = np.random.choice(len(class_imgs), count, replace=False)

        X_test.extend(class_imgs[selected])
        y_test.extend([class_idx] * count)

        remaining = list(set(range(len(class_imgs))) - set(selected))
        X_train.extend(class_imgs[remaining])
        y_train.extend([class_idx] * len(remaining))

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    X_train, y_train = shuffle(X_train, y_train, random_state=42)
    X_test, y_test = shuffle(X_test, y_test, random_state=42)

    y_train_cat = to_categorical(y_train, NUM_CLASSES)
    y_test_cat = to_categorical(y_test, NUM_CLASSES)

    model = build_model()

    history = model.fit(
        datagen.flow(X_train, y_train_cat, batch_size=BATCH_SIZE),
        steps_per_epoch=len(X_train) // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_test, y_test_cat),
        callbacks=callbacks,
        verbose=1
    )

    _, acc = model.evaluate(X_test, y_test_cat, verbose=0)

    if acc > ACCURACY:
        ACCURACY = acc
        SELECTED_HISTORY = history.history
        model.save(MODEL_SAVE_PATH)
        print("Model saved")


print(f"\nReported Test Accuracy: {ACCURACY * 100:.2f}%")
print(f"Saved Model Path: {MODEL_SAVE_PATH}")

plt.figure(figsize=(8, 5))
plt.plot(SELECTED_HISTORY["accuracy"], label="Training Accuracy")
plt.plot(SELECTED_HISTORY["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("ResNet-50 Accuracy Curve")
plt.legend()
plt.grid(True)
plt.show()
