from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from model_final.model_loader import SentimentModel
from logging import getLogger, INFO
from azure.monitor.opentelemetry import configure_azure_monitor

app = FastAPI()
model = SentimentModel()  # chargement unique au démarrage

configure_azure_monitor(
    connection_string="InstrumentationKey=82b3e923-352b-492b-87cb-91cd5d07c9c4",  # ou ApplicationInsightsConnectionString=...
)
logger = getLogger(__name__)
logger.setLevel(INFO)


# --- Schémas de données ---
class TextInput(BaseModel):
    text: str

# Feedback après la prédiction
class FeedbackRequest(BaseModel):
    texte: str
    prediction: int
    feedback_correct: bool


# --- Route d'inférence ---
@app.post("/predict")
def predict(input: TextInput):
    texts = [input.text]
    prediction = int(model.predict(texts)[0])
    return {
        "text": input.text,
        "prediction": prediction
    }

@app.post("/feedback")
def feedback(data: FeedbackRequest):
    logger.info("FeedbackLog", extra={

            "texte": data.texte,
            "prediction": data.prediction,
            "correct": data.feedback_correct
        
    })
    return {"message": "Feedback envoyé à App Insights"}
