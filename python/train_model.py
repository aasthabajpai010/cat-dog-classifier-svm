import os
import cv2
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

dataset_path = "../dataset"

categories = ["cat", "dog"]

data = []

labels = []

# -------------------------------------------------
# LOAD IMAGES
# -------------------------------------------------

for category in categories:

    folder_path = os.path.join(
        dataset_path,
        category
    )

    label = categories.index(category)

    for image_name in os.listdir(folder_path):

        image_path = os.path.join(
            folder_path,
            image_name
        )

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
            (64,64)
        )

        # Flatten image

        flattened = resized.flatten()

        data.append(flattened)

        labels.append(label)

# -------------------------------------------------
# NUMPY ARRAY
# -------------------------------------------------

data = np.array(data)

labels = np.array(labels)

# -------------------------------------------------
# SPLIT DATASET
# -------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    data,
    labels,
    test_size=0.30,
    random_state=42
)

# -------------------------------------------------
# TRAIN MODEL
# -------------------------------------------------

model = SVC(
    kernel='linear',
    probability=True
)

model.fit(
    X_train,
    y_train
)

# -------------------------------------------------
# ACCURACY
# -------------------------------------------------

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\nModel Accuracy:", accuracy*100)

# -------------------------------------------------
# SAVE MODEL
# -------------------------------------------------

joblib.dump(
    model,
    "svm_model.pkl"
)

print("\nModel Saved Successfully")