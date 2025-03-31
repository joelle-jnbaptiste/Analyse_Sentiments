![Deploy to Azure](https://github.com/joelle-jnbaptiste/Analyse_Sentiments/actions/workflows/deploy.yml/badge.svg)

# Lancer l'environnement:
.\.env-project\Scripts\activate

# Lancer le serveur MLFLOW
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000

# Lancer l'API
cd api\ 
uvicorn main:app --reload 

# Doc Swagger
http://127.0.0.1:8000/docs#/

#Docker
## build l'image
docker build -t myapi-mlflow .


#Lancer le container
docker exec -p 8000:8000 myapi-mlflow

docker start -a sentiment-multiapp

