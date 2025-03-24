# Lancer l'environnement:
.\.env-project\Scripts\activate

# Lancer le serveur MLFLOW
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000

# Lancer l'API
cd .\api\ 
uvicorn main:app --reload 

# Doc Swagger
http://127.0.0.1:8000/docs#/