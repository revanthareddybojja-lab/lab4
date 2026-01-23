import pandas as pd
import joblib
import json
import os
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("dataset/winequality-red.csv", sep=";")

X = df.drop("quality", axis=1)   # 11 features
y = df["quality"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.pkl")

# Save metrics
with open("metrics.json", "w") as f:
    json.dump({"accuracy": acc}, f)

print("Training completed. Accuracy:", acc)
