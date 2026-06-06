import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

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

# Accuracy

accuracy = accuracy_score(
    y_test,
    predictions
)

print("ACCURACY:", accuracy)

# Confusion matrix

cm = confusion_matrix(
    y_test,
    predictions
)

print("\nCONFUSION MATRIX\n")

print(cm)

# Classification report

report = classification_report(
    y_test,
    predictions
)

print("\nCLASSIFICATION REPORT\n")

print(report)

# Visualization

plt.figure(figsize=(6,6))

plt.imshow(
    cm,
    cmap='Blues'
)

plt.title("Confusion Matrix")

plt.colorbar()

plt.xticks([0,1], ["Cat","Dog"])

plt.yticks([0,1], ["Cat","Dog"])

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.show()