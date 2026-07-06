# MyTwin Avatar — backend API (conteneur serverless Scaleway).
FROM python:3.12-slim

# Libs système requises par opencv (tiré par mediapipe) au runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    U2NET_HOME=/opt/models/u2net

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pré-télécharge le modèle rembg (u2net_human_seg) dans l'image pour un cold
# start fini (sinon ~170 Mo téléchargés au 1er /body).
RUN python -c "from rembg import new_session; new_session('u2net_human_seg')"

# Pipeline + modèle FaceLandmarker + corps/visage de test.
COPY pipeline/ ./pipeline/
COPY models/ ./models/
COPY corps.glb visage.glb api.py ./

# Meshy réel par défaut en prod ; passer USE_LOCAL_BODY/FACE=1 pour la démo locale.
ENV USE_LOCAL_BODY=0 \
    USE_LOCAL_FACE=0

EXPOSE 8080
# timeout large : /graft est synchrone (CPU).
CMD gunicorn api:app --workers 1 --threads 8 --timeout 300 --bind 0.0.0.0:${PORT:-8080}
