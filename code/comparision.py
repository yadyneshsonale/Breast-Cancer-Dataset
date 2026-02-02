import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils import resample
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.applications import VGG16, InceptionV3
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array


# ===================== PATHS =====================
image_folder_left = "./Breast-Cancer-Dataset/paper_data/left"
label_file_left = "./Breast-Cancer-Dataset/paper_data/left/left.xlsx"

image_folder_right = "./Breast-Cancer-Dataset/paper_data/right"
label_file_right = "./Breast-Cancer-Dataset/paper_data/right/right.xlsx"

output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

IMG_SIZE = (224, 224)
NUM_RUNS = 3


# ===================== IMAGE PREPROCESS =====================
def preprocess_image(image_path, model_name):
    image = load_img(image_path, target_size=IMG_SIZE, color_mode="rgb")
    image = img_to_array(image)

    if model_name == "vgg16":
        image = vgg_preprocess(image)
    elif model_name == "inceptionv3":
        image = inception_preprocess(image)

    return image


# ===================== DATASET LOADERS =====================
def process_dataset(image_folder, label_file, model_name, is_right=False):
    labels_df = pd.read_excel(label_file)
    labels_df.set_index("Image", inplace=True)
    labels_df.index = labels_df.index.astype(str)

    X, y = [], []
    for image_name in os.listdir(image_folder):
        if image_name.endswith(".png"):
            base = os.path.splitext(image_name)[0]
            if is_right:
                base = f"{base}_right"

            if base in labels_df.index:
                img_path = os.path.join(image_folder, image_name)
                img = preprocess_image(img_path, model_name)
                X.append(img)
                y.append(labels_df.loc[base, "Label"])

    return np.array(X), np.array(y)


# ===================== MODEL FACTORY =====================
def create_model(model_name, num_classes):
    if model_name == "vgg16":
        base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
        freeze_until = 15

    elif model_name == "inceptionv3":
        base_model = InceptionV3(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
        freeze_until = 249

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax")
    ])

    base_model.trainable = True
    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ===================== TRAINING LOOP =====================
models_to_train = ["vgg16", "inceptionv3"]

for model_name in models_to_train:
    print(f"\n🚀 Training {model_name.upper()}")

    # Load data
    X_left, y_left = process_dataset(image_folder_left, label_file_left, model_name)
    X_right, y_right = process_dataset(image_folder_right, label_file_right, model_name, is_right=True)

    X = np.vstack((X_left, X_right))
    y = np.concatenate((y_left, y_right))

    # Encode labels
    class_mapping = {label: idx for idx, label in enumerate(np.unique(y))}
    y = np.array([class_mapping[label] for label in y])
    y = to_categorical(y, num_classes=len(class_mapping))

    # ===================== RESAMPLING =====================
    X_res, y_res = [], []
    max_count = max(Counter(np.argmax(y, axis=1)).values())

    for cls in range(len(class_mapping)):
        idx = np.argmax(y, axis=1) == cls
        X_cls, y_cls = X[idx], y[idx]

        X_up, y_up = resample(
            X_cls, y_cls,
            replace=True,
            n_samples=max_count,
            random_state=42
        )
        X_res.append(X_up)
        y_res.append(y_up)

    X_resampled = np.vstack(X_res)
    y_resampled = np.vstack(y_res)

    # Shuffle
    perm = np.random.permutation(len(X_resampled))
    X_resampled = X_resampled[perm]
    y_resampled = y_resampled[perm]

    # ===================== RUNS =====================
    for run in range(NUM_RUNS):
        print(f"\nRun {run + 1}/{NUM_RUNS}")

        X_train, X_val, y_train, y_val = train_test_split(
            X_resampled,
            y_resampled,
            test_size=0.2,
            random_state=42 + run,
            stratify=y_resampled
        )

        # Class weights
        y_train_int = np.argmax(y_train, axis=1)
        weights = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(y_train_int),
            y=y_train_int
        )
        class_weights = dict(enumerate(weights))

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
            ModelCheckpoint(
                filepath=f"{model_name}_best_run_{run}.keras",
                monitor="val_accuracy",
                save_best_only=True
            ),
            ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=5, min_lr=1e-6)
        ]

        datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode="nearest"
        )
        datagen.fit(X_train)

        model = create_model(model_name, num_classes=len(class_mapping))

        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=16),
            epochs=70,
            validation_data=(X_val, y_val),
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=0
        )

        # ===================== CONFUSION MATRIX =====================
        y_pred = np.argmax(model.predict(X_val), axis=1)
        y_true = np.argmax(y_val, axis=1)

        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=list(class_mapping.keys()))
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f"{model_name.upper()} – Run {run + 1}")
        plt.savefig(f"{output_dir}/{model_name}_cm_run_{run + 1}.png")
        plt.close()
