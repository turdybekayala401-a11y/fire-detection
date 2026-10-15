import pandas as pd
import numpy as np
import string

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv("dataset.csv")

data = data.dropna(subset=["text_", "label"])

X = data["text_"]
y = data["label"]
 
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        max_features=5000
    )),
    ("classifier", RandomForestClassifier(
        n_estimators=30,
        random_state=42,
        n_jobs=-1
    ))
])

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Fake Review Detection System")
print("-" * 40)
print(f"Model accuracy: {accuracy * 100:.2f}%")
print()
print("Enter a product review.")
print("Type 'exit' to stop.")

while True:
    review = input("\nReview: ")

    if review.lower() == "exit":
        print("Program stopped.")
        break

    result = model.predict([review])[0]

    if result == "OR":
        print("Result: FAKE REVIEW")
    elif result == "CG":
        print("Result: REAL REVIEW")
    else:
        print(f"Result: {result}")