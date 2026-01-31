import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, auc

# Your previous functions
def numbers(image):
    d = np.zeros(256)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            pixel_value = image[i][j]
            d[pixel_value] += 1
    return d

def subs(arr1, arr2):
    d = np.zeros(256)
    for i in range(256):
        d[i] = (arr1[i] + arr2[i]) / 100
    return d

# Dataset path
dataset_path = "../symmetry data"

database_b = []
database_m = []

# Load images into database_b and database_m
for i in range(1, 120):
    path = os.path.join(dataset_path, f"p{i}")
    if os.path.exists(path):
        if os.path.exists(os.path.join(path, "b1.jpg")):
            img1 = cv2.imread(os.path.join(path, "b1.jpg"), cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(os.path.join(path, "b2.jpg"), cv2.IMREAD_GRAYSCALE)
            database_b.append(subs(numbers(img1), numbers(img2)))
        if os.path.exists(os.path.join(path, "m1.jpg")):
            img1 = cv2.imread(os.path.join(path, "m1.jpg"), cv2.IMREAD_GRAYSCALE)
            img2 = cv2.imread(os.path.join(path, "m2.jpg"), cv2.IMREAD_GRAYSCALE)
            database_m.append(subs(numbers(img1), numbers(img2)))

# Label mapping
categories = {"PB": 0, "PM": 1}
file_paths = {"PB": database_b, "PM": database_m}

# Prepare data and labels
data = []
labels = []

for category, paths in file_paths.items():
    for d in paths:
        data.append(d)
        labels.append(categories[category])

data = np.array(data)
labels = np.array(labels)

# Shuffle data
data, labels = shuffle(data, labels, random_state=42)

# One-hot encode labels
encoder = OneHotEncoder(sparse_output=False)
labels_one_hot = encoder.fit_transform(labels.reshape(-1, 1))

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(data, labels_one_hot, test_size=0.3, random_state=42)

# Define the model
def create_model():
    inputs = Input(shape=(256,))
    x = Dense(256, activation='relu')(inputs)
    x = Dropout(0.4)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.4)(x)
    output = Dense(2, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=Adam(learning_rate=0.01), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    return model

# Create the model
model = create_model()

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32,
    verbose=1
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Plot accuracy
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Plot loss
plt.subplot(1, 3, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# ROC Curve
plt.subplot(1, 3, 3)
y_pred_prob = model.predict(X_test)
fpr, tpr, _ = roc_curve(y_test[:, 1], y_pred_prob[:, 1])
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.grid(True)

plt.tight_layout()
plt.show()

# Print model summary
print("\nModel Summary:")
model.summary()