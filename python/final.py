import matplotlib.pyplot as plt

accuracy = 92

precision = 91

recall = 90

f1 = 90

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score"
]

values = [
    accuracy,
    precision,
    recall,
    f1
]

plt.figure(figsize=(8,6))

plt.bar(
    metrics,
    values
)

plt.ylim(0,100)

plt.title("SVM Model Performance")

plt.ylabel("Percentage")

plt.show()