\# Cat vs Dog Image Classifier using SVM



\## Project Overview



This project is a machine learning based image classification system that classifies images as either a Cat or a Dog using a Support Vector Machine (SVM) classifier.



The application provides a simple Streamlit interface where users can upload an image and receive a prediction along with the confidence score.



\---



\## Features



\* Upload cat or dog images

\* Image preprocessing using OpenCV

\* Classification using Support Vector Machine (SVM)

\* Confidence score display

\* User-friendly Streamlit interface



\---



\## Technologies Used



\* Python

\* OpenCV

\* NumPy

\* Scikit-Learn

\* Streamlit

\* Joblib

\* Matplotlib



\---



\## Project Structure



AI\_Project\_SVM/



├── python/



│ ├── app.py



│ ├── preprocess.py



│ ├── train\_model.py



│ ├── train\_svm.py



│ ├── predict.py



│ └── evaluate.py



├── requirements.txt



├── .gitignore



└── README.md



\---



\## Dataset



The project uses cat and dog images for binary image classification.



Note: The dataset is not included in this repository due to its large size.



\---



\## Installation



```bash

pip install -r requirements.txt

```



\## Run the Application



```bash

streamlit run python/app.py

```



\---



\## Workflow



1\. Load image dataset

2\. Preprocess images using OpenCV

3\. Train SVM classifier

4\. Save trained model

5\. Upload image through Streamlit

6\. Predict Cat or Dog

7\. Display confidence score



\---



\## Future Improvements



\* Improve model accuracy using feature extraction techniques

\* Add support for multiple animal classes

\* Deploy the application online

\* Compare SVM with deep learning models



\---



\## Author



Aastha Bajpai

MCA Student, NIT Jamshedpur



