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

# Train model

model = SVC(kernel='linear')

model.fit(X_train, y_train)

# Predictions

predictions = model.predict(X_test)

# Visualization

plt.figure(figsize=(12,8))

for i in range(6):

    plt.subplot(2,3,i+1)

    image = X_test[i].reshape(64,64)

    plt.imshow(
        image,
        cmap='gray'
    )

    predicted_label = categories[predictions[i]]

    actual_label = categories[y_test[i]]

    plt.title(
        f"P:{predicted_label}\nA:{actual_label}"
    )

    plt.axis("off")

plt.suptitle("Predicted vs Actual")

plt.show()