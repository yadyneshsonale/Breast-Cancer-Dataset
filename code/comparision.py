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


# ===================== CONFIG =====================
IMG_SIZE = (224, 224)
NUM_RUNS = 3
BATCH_SIZE = 16
EPOCHS = 70
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

image_folder_left = "./Breast-Cancer-Dataset/paper_data/left"
label_file_left = "./Breast-Cancer-Dataset/paper_data/left/left.xlsx"
image_folder_right = "./Breast-Cancer-Dataset/paper_data/right"
label_file_right = "./Breast-Cancer-Dataset/paper_data/right/right.xlsx"


# ===================== PREPROCESS =====================
def preprocess_image(image_path, model_name):
    img = load_img(image_path, target_size=IMG_SIZE, color_mode="rgb")
    img = img_to_array(img)

    if model_name == "vgg16":
        img = vgg_preprocess(img)
    elif model_name == "inceptionv3":
        img = inception_preprocess(img)

    return img


# ===================== DATA LOADER =====================
def process_dataset(image_folder, label_file, model_name, is_right=False):
    df = pd.read_excel(label_file)
    df.set_index("Image", inplace=True)
    df.index = df.index.astype(str)

    X, y = [], []
    for fname in os.listdir(image_folder):
        if fname.endswith(".png"):
            key = os.path.splitext(fname)[0]
            if is_right:
                key = f"{key}_right"

            if key in df.index:
                img = preprocess_image(os.path.join(image_folder, fname), model_name)
                X.append(img)
                y.append(df.loc[key, "Label"])

    return np.array(X), np.array(y)


# ===================== MODEL =====================
def create_model(model_name, num_classes):
    if model_name == "vgg16":
        base = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
        freeze_until = 15
    else:
        base = InceptionV3(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
        freeze_until = 249

    model = Sequential([
        base,
        GlobalAveragePooling2D(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax")
    ])

    for layer in base.layers[:freeze_until]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ===================== TRAIN =====================
for model_name in ["vgg16", "inceptionv3"]:
    print(f"\n🚀 Training {model_name.upper()}")

    Xl, yl = process_dataset(image_folder_left, label_file_left, model_name)
    Xr, yr = process_dataset(image_folder_right, label_file_right, model_name, True)

    X = np.vstack((Xl, Xr))
    y = np.concatenate((yl, yr))

    class_map = {l: i for i, l in enumerate(np.unique(y))}
    y = to_categorical([class_map[v] for v in y], len(class_map))

    # -------- Resample --------
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
    X_res = np.vstack(X_resampled)
    y_res = np.vstack(y_resampled)

    perm = np.random.permutation(len(X_res))
    X_res, y_res = X_res[perm], y_res[perm]

    best_val_acc = -np.inf
    best_model = None
    best_X_val = None
    best_y_val = None
    best_run = None

    for run in range(NUM_RUNS):
        print(f"\nRun {run + 1}/{NUM_RUNS}")

        X_train, X_val, y_train, y_val = train_test_split(
            X_res, y_res,
            test_size=0.2,
            random_state=42 + run,
            stratify=y_res
        )

        y_int = np.argmax(y_train, axis=1)
        cw = dict(enumerate(
            compute_class_weight("balanced", np.unique(y_int), y_int)
        ))

        datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
        )
        datagen.fit(X_train)

        model = create_model(model_name, len(class_map))

        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
            epochs=EPOCHS,
            validation_data=(X_val, y_val),
            class_weight=cw,
            callbacks=[
                EarlyStopping(patience=10, restore_best_weights=True),
                ReduceLROnPlateau(patience=5, factor=0.3, min_lr=1e-6)
            ],
            verbose=1
        )

        run_best = max(history.history["val_accuracy"])

        if run_best > best_val_acc:
            best_val_acc = run_best
            best_model = model
            best_X_val = X_val
            best_y_val = y_val
            best_run = run + 1

    print(f"\nRUN for {model_name.upper()}")
    print(f"VAL ACCURACY: {best_val_acc:.4f}")

    y_pred = np.argmax(best_model.predict(best_X_val), axis=1)
    y_true = np.argmax(best_y_val, axis=1)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=list(class_map.keys()))
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"{model_name.upper()} – Best Run {best_run}")
    plt.savefig(f"{output_dir}/{model_name}_BEST_CM.png")
    plt.show()
