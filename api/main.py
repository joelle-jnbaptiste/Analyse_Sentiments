from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
from mlflow.pyfunc import load_model
import pandas as pd
import uuid
from datetime import datetime
from mlflow.tracking import MlflowClient


app = FastAPI()



# Configurer MLflow pour utiliser ce dossier
mlflow.set_tracking_uri("http://127.0.0.1:5000")

client = MlflowClient()

# 1. Trouver l'ID de l'expérience par son nom
experiment = client.get_experiment_by_name("AnalyseSentiments")
if experiment is None:
    raise RuntimeError(" L'expérience n'existe pas.")
experiment_id = experiment.experiment_id

# 2. Récupérer les runs triés par date de début (descendant)
runs = client.search_runs(
    experiment_ids=[experiment_id],
    order_by=["start_time DESC"],
    max_results=1
)

if not runs:
    raise RuntimeError(" Aucun run trouvé dans cette expérience.")

# 3. Récupérer le run_id
run_id = runs[0].info.run_id
print(f" Dernier run_id : {run_id}")

MODEL_URI = f"runs:/{run_id}/model"


model = load_model(MODEL_URI)

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
    # Le modèle attend un DataFrame avec une colonne texte
    df = pd.DataFrame([{"text": input.text}])  # colonne = nom attendu par ton modèle de prétraitement

    prediction = int(model.predict(df)[0])

    return {
        "text": input.text,
        "prediction": prediction
    }


# --- Route de feedback ---
@app.post("/feedback")
def feedback(input: FeedbackInput):
    experiment_name = "AnalyseSentiments"
    mlflow.set_experiment(experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    run_name = f"feedback-{uuid.uuid4()}"
    with mlflow.start_run(experiment_id=experiment.experiment_id,run_name=run_name, nested=True):
        mlflow.log_param("text", input.text)
        mlflow.log_param("prediction", input.prediction)
        mlflow.log_param("correct", input.correct)
        mlflow.set_tag("feedback", "user_feedback")
        if not input.correct:
            mlflow.set_tag("status", "wrong_prediction")
            mlflow.log_metric("wrong", 1)
        else:
            mlflow.log_metric("correct", 1)
    return {
        "message": "Feedback reçu et loggé dans MLflow",
        "correct": input.correct
    }
