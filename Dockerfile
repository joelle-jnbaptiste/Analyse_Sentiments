FROM python:3.9-slim

# Installer supervisord, nginx et les outils nécessaires
RUN apt-get update && apt-get clean && apt-get install -y ca-certificates  git-lfs && git lfs install

# Définir le répertoire de travail
RUN git clone --branch ton-branche https://github.com/joelle-jnbaptiste/Analyse_Sentiments.git /app  

WORKDIR /app

RUN git lfs pull

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements-api.txt


# Exposer uniquement le port que Nginx écoutera (Azure ne gère qu'un port sortant)
EXPOSE 8000

# 4. Lancer l'application FastAPI avec Uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4" ]