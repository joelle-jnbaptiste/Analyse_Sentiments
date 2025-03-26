FROM python:3.9-slim

# Installer supervisord, nginx et les outils nécessaires
RUN apt-get update && apt-get install -y nginx curl && apt-get clean

# Définir le répertoire de travail
WORKDIR /app

# Copier les requirements
COPY ./requirements-api.txt /app/requirements-api.txt

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements-api.txt
RUN pip install mlflow

# Copier les modèles
COPY model_final /app/model_final/

# Copier le code de l'application
COPY . /app

# Copier la conf de supervisord
COPY ./start.sh /app/start.sh

# Copier la conf nginx
COPY ./nginx.conf /etc/nginx/sites-enabled/default


# Exposer uniquement le port que Nginx écoutera (Azure ne gère qu'un port sortant)
EXPOSE 80

# Lancer supervisord pour tout gérer (API + MLflow + Nginx)
RUN chmod +x /app/start.sh
RUN ls -l /app
CMD ["/app/start.sh"]
