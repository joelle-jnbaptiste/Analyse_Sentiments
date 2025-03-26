#!/bin/bash
cd /app
rm -rf mlruns/

echo "Démarrage du serveur MLflow..."
mlflow server --backend-store-uri /app/mlruns --host 0.0.0.0 --port 5000 &

# Attendre que MLflow soit prêt
echo "En attente que MLflow soit prêt..."
for i in {1..60}; do
    curl -s http://127.0.0.1:5000 > /dev/null && break
    echo " Tentative $i : MLflow pas encore prêt..."
    sleep 1
done

echo "MLflow prêt, lancement du script de log..."
cd /app/model_final
python log_model_after_mlflow.py

echo " Lancement de l'API FastAPI..."
cd /app/api
uvicorn main:app --host 0.0.0.0 --port 8000 --root-path /api &

echo "Démarrage de Nginx..."
nginx -g "daemon off;" 


