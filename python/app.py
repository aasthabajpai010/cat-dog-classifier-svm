import streamlit as st
import cv2
import numpy as np
import joblib

from PIL import Image

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------

st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐶",
    layout="centered"
)

# -------------------------------------------------
# TITLE
# -------------------------------------------------

st.title("🐱 Cat vs 🐶 Dog Classifier")

st.subheader(
    "Image Classification using SVM"
)

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------

model = joblib.load("svm_model.pkl")

categories = ["cat", "dog"]

# -------------------------------------------------
# FILE UPLOADER
# -------------------------------------------------

# -------------------------------------------------
# FILE UPLOADER
# -------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload Cat or Dog Image",
    type=["jpg", "jpeg", "png"]
)

# -------------------------------------------------
# PREDICTION SECTION
# -------------------------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        width=300
    )

    if st.button("Predict Image"):

        # Convert image to numpy array

        image_array = np.array(image)

        # Convert to grayscale

        gray = cv2.cvtColor(
            image_array,
            cv2.COLOR_BGR2GRAY
        )

        # Resize image

        resized = cv2.resize(
            gray,
            (64,64)
        )

        # Flatten image

        flattened = resized.flatten()

        # Reshape image

        test_data = flattened.reshape(1,-1)

        # Predict class

        prediction = model.predict(test_data)

        # Predict probability

        probability = model.predict_proba(test_data)

        # Confidence

        confidence = np.max(probability) * 100

        # Predicted class

        predicted_class = categories[prediction[0]]

        # -------------------------------------------------
        # RESULT DISPLAY
        # -------------------------------------------------

        if predicted_class == "cat":

            st.success("🐱 Prediction: CAT")

        else:

            st.success("🐶 Prediction: DOG")

        st.info(
            f"Confidence: {confidence:.2f}%"
        )

        # Low confidence warning

        if confidence < 55:

            st.warning(
                "⚠ Model confidence is low."
            )