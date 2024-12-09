#!/usr/bin/env bash
#https://www.youtube.com/watch?v=NPmNCu1L7uw

# Állítsd be ezeket a változókat saját projektednek megfelelően
PROJECT_ID="sas-ollama"
REGION="us-central1"      # Egy olyan régió, ahol a GPU elérhető Cloud Run-on
SERVICE_NAME="ollama-gpu-service"
IMAGE_NAME="ollama-gpu"
ARTIFACT_REPO="us-central1-docker.pkg.dev/${PROJECT_ID}/ollama-repo/${IMAGE_NAME}"

# Config beállítás
gcloud config set project $PROJECT_ID
gcloud config set run/region $REGION

# Ha még nincs Artifact Registry repo létrehozva, hozzuk létre
# (Ha már létezik, ez a lépés kihagyható)
gcloud artifacts repositories create ollama-repo \
  --repository-format=docker \
  --location=$REGION

# Docker image build és push
# Ha lokális Dockerfile-ból akarsz építeni:
gcloud builds submit --tag $ARTIFACT_REPO

# Cloud Run telepítés GPU-val
# Feltételezve, hogy a Docker image GPU-kompatibilis, és Ollama fut benne.
gcloud run deploy $SERVICE_NAME \
  --image $ARTIFACT_REPO \
  --region $REGION \
  --allow-unauthenticated \
  --cpu=4 \
  --memory=16Gi \
  --gpu=1 \
  --port=1140 \
  --platform=managed

echo "A szolgáltatás sikeresen telepítve. A Cloud Run URL megtekintéséhez futtasd:"
echo "gcloud run services describe $SERVICE_NAME --region=$REGION --format='value(status.url)'"

