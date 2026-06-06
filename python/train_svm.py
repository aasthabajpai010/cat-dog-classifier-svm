import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC

dataset_path = "dataset"

data = []
labels = []

categories = ["cat", "dog"]

# Load images

for category in categories:

    folder_path = os.path.join(dataset_path, category)

    label = categories.index(category)

    for image_name in os.listdir(folder_path):

        image_path = os.path.join(folder_path, image_name)

        image = cv2.imread(image_path)

        if image is None:
            continue

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        resized = cv2.resize(
            gray,
            (64,64)
        )

        flattened = resized.flatten()

        data.append(flattened)

        labels.append(label)

data = np.array(data)

labels = np.array(labels)

# Split dataset

X_train, X_test, y_train, y_test = train_test_split(
    data,
    labels,
    test_size=0.30,
    random_state=42
)

# Train SVM model

model = SVC(kernel='linear')

model.fit(X_train, y_train)

print("SVM MODEL TRAINED SUCCESSFULLY")
# ==============================
# Training Accuracy
# ==============================

train_accuracy = model.score(X_train, y_train)

print("Training Accuracy :", train_accuracy)

# ==============================
# Testing Accuracy
# ==============================

test_accuracy = model.score(X_test, y_test)

print("Testing Accuracy :", test_accuracy)

# Visualization

cat_count = np.sum(labels == 0)

dog_count = np.sum(labels == 1)

plt.figure(figsize=(6,5))

plt.bar(
    ["Cat", "Dog"],
    [cat_count, dog_count]
)

plt.title("Dataset Distribution")

plt.ylabel("Number of Images")

plt.show()