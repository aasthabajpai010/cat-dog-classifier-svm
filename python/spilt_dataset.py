import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

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

print("TRAINING DATA:", X_train.shape)

print("TESTING DATA:", X_test.shape)

# Visualization

sizes = [len(X_train), len(X_test)]

labels_name = ["Training", "Testing"]

plt.figure(figsize=(6,6))

plt.pie(
    sizes,
    labels=labels_name,
    autopct='%1.1f%%'
)

plt.title("Dataset Split")

plt.show()