FROM python:3.9-slim

# Installer supervisord, nginx et les outils nécessaires
RUN apt-get update && apt-get clean

# Définir le répertoire de travail
WORKDIR /app

# Copier les requirements
COPY ./requirements-api.txt /app/requirements-api.txt

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements-api.txt

# Copier les modèles
COPY model_final /app/model_final/

# Copier le code de l'application
COPY api /app/api/

# Exposer uniquement le port que Nginx écoutera (Azure ne gère qu'un port sortant)
EXPOSE 8000

# 4. Lancer l'application FastAPI avec Uvicorn
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]