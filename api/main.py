from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from logging import getLogger, INFO

from model_final.model_loader import SentimentModel
from azure.monitor.opentelemetry import configure_azure_monitor

# Initialize the FastAPI app
app = FastAPI()

# Load the sentiment analysis model
model = SentimentModel()

# Configure Azure Monitor (Application Insights)
configure_azure_monitor(
    connection_string="InstrumentationKey=82b3e923-352b-492b-87cb-91cd5d07c9c4"
)

# Initialize the logger
logger = getLogger(__name__)
logger.setLevel(INFO)


# Request model for prediction endpoint
class TextInput(BaseModel):
    text: str


# Request model for feedback endpoint
class FeedbackRequest(BaseModel):
    texte: str
    prediction: int
    feedback_correct: bool


@app.post("/predict")
def predict(input: TextInput):
    """
    Predict the sentiment of the input text.
    Returns a dictionary with the input text and its prediction (0 or 1).
    """
    texts = [input.text]
    prediction = int(model.predict(texts)[0])

    return {
        "text": input.text,
        "prediction": prediction
    }


@app.post("/feedback")
def feedback(data: FeedbackRequest):
    """
    Receive feedback about the prediction result.
    Logs the input text, prediction, and user feedback to Azure Monitor.
    """
    logger.info(
        "FeedbackLog",
        extra={
            "texte": data.texte,
            "prediction": data.prediction,
            "correct": data.feedback_correct
        }
    )
    return {"message": "Feedback sent to App Insights"}
