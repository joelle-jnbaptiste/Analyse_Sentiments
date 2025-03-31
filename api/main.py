from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from model_final.model_loader import SentimentModel


app = FastAPI()
model = SentimentModel()  # chargement unique au démarrage

# --- Schémas de données ---
class TextInput(BaseModel):
    text: str

class FeedbackInput(BaseModel):
    text: str
    prediction: int
    correct: bool


# --- Route d'inférence ---
@app.post("/predict")
def predict(input: TextInput):
    texts = [input.text]
    prediction = int(model.predict(texts)[0])
    return {
        "text": input.text,
        "prediction": prediction
    }
