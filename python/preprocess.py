import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Dataset path
dataset_path = "dataset"

data = []
labels = []

categories = ["cat", "dog"]

# -------------------------------------------------
# LOAD AND PREPROCESS IMAGES
# -------------------------------------------------

for category in categories:

    folder_path = os.path.join(dataset_path, category)

    label = categories.index(category)

    for image_name in os.listdir(folder_path):

        image_path = os.path.join(folder_path, image_name)

        image = cv2.imread(image_path)

        if image is None:
            continue

        # Convert to grayscale

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        # Resize image

        resized = cv2.resize(
            gray,
            (64, 64)
        )

        # Flatten image

        flattened = resized.flatten()

        data.append(flattened)

        labels.append(label)

# Convert into numpy arrays

data = np.array(data)

labels = np.array(labels)

print("DATA SHAPE:", data.shape)

print("LABEL SHAPE:", labels.shape)

# -------------------------------------------------
# VISUALIZATION
# -------------------------------------------------

plt.figure(figsize=(10,5))

for i in range(6):

    plt.subplot(2,3,i+1)

    image = data[i].reshape(64,64)

    plt.imshow(
        image,
        cmap='gray'
    )

    plt.title(
        categories[labels[i]]
    )

    plt.axis("off")

plt.suptitle("Preprocessed Images")

plt.show()

print("PREPROCESSING COMPLETED")