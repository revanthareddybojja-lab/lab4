from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

# ✅ Correct path for Docker & CI/CD
model = joblib.load("models/model.pkl")

class WineInput(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float

@app.post("/predict")
def predict(data: WineInput):
    features = [[
        data.fixed_acidity,
        data.volatile_acidity,
        data.citric_acid,
        data.residual_sugar,
        data.chlorides,
        data.free_sulfur_dioxide,
        data.total_sulfur_dioxide,
        data.density,
        data.pH,
        data.sulphates,
        data.alcohol
    ]]

    prediction = model.predict(features)

    return {
        "name": "Revanth Reddy",
        "roll_no": "2022BCS0210",
        "wine_quality": int(prediction[0])
    }
