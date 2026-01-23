import json
import joblib
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X = np.array([[1], [2], [3], [4]])
y = np.array([0, 0, 1, 1])

model = LogisticRegression()
model.fit(X, y)

preds = model.predict(X)
acc = accuracy_score(y, preds)

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.pkl")

with open("metrics.json", "w") as f:
    json.dump({"accuracy": acc}, f)

print("Training done | Accuracy:", acc)
