import time
from distilbert_pyfunc import DistilBertPyFunc
import mlflow.pyfunc
import os
# Se placer dans le dossier contenant les artefacts
os.chdir(os.path.dirname(__file__))


mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("AnalyseSentiments")

with mlflow.start_run() as run:
    run_id = run.info.run_id
    experiment_id = run.info.experiment_id
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=DistilBertPyFunc(),
        code_path=["distilbert_pyfunc.py"],
        artifacts={"state_dict": "DISTILBERT_MODEL_FULLY_TRAINED.pt"},
        registered_model_name="sentiment-model"
    )
    print("Modèle loggé dans MLflow !")

    # Enregistrer le run_id dans un fichier lisible par l'API
    with open("/app/mlflow_run_info.txt", "w") as f:
        f.write(f"{experiment_id}\n{run_id}")