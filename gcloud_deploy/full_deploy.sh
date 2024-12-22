#!/usr/bin/env bash

# Projekt és régió beállítása
PROJECT_ID="sas-ollama"
REGION="us-central1"
ARTIFACT_REPO_NAME="ollama-repo"
SERVICE_NAME_OLLAMA="ollama-gpu-service"
SERVICE_NAME_NEO4J="neo4j-vector-service"
SERVICE_NAME_APP="chatbot-app-service"
OLLAMA_IMAGE_NAME="ollama-gpu"
NEO4J_IMAGE_NAME="neo4j-service"
APP_IMAGE_NAME="chatbot-app"

# Artifact Registry URL-ek
ARTIFACT_REPO="${REGION}-docker.pkg.dev/${PROJECT_ID}/${ARTIFACT_REPO_NAME}"

# GCloud Config beállítása
echo "GCloud projekt és régió beállítása..."
gcloud config set project $PROJECT_ID
gcloud config set run/region $REGION

# Artifact Registry létrehozása (ha nem létezik)
if ! gcloud artifacts repositories describe $ARTIFACT_REPO_NAME --location=$REGION; then
    echo "Artifact Registry létrehozása..."
    gcloud artifacts repositories create $ARTIFACT_REPO_NAME \
        --repository-format=docker \
        --location=$REGION
fi

# Ollama service Docker image build és push
echo "Ollama Docker image build és push..."
gcloud builds submit --tag ${ARTIFACT_REPO}/${OLLAMA_IMAGE_NAME} .

# Neo4j service Docker image build és push
echo "Neo4j Docker image build és push..."
gcloud builds submit --tag ${ARTIFACT_REPO}/${NEO4J_IMAGE_NAME} .

# App service Docker image build és push
echo "Chatbot App Docker image build és push..."
gcloud builds submit --tag ${ARTIFACT_REPO}/${APP_IMAGE_NAME} .

# Ollama Service deploy Cloud Run-ra GPU-val (Beta)
echo "Ollama GPU service deploy..."
gcloud beta run deploy $SERVICE_NAME_OLLAMA \
    --image ${ARTIFACT_REPO}/${OLLAMA_IMAGE_NAME} \
    --allow-unauthenticated \
    --cpu=4 --memory=16Gi \
    --gpu=1 --port=11434 \
    --platform=managed

# Neo4j Service deploy Cloud Run-ra
echo "Neo4j service deploy..."
gcloud run deploy $SERVICE_NAME_NEO4J \
    --image ${ARTIFACT_REPO}/${NEO4J_IMAGE_NAME} \
    --allow-unauthenticated \
    --cpu=2 --memory=8Gi \
    --port=7474 \
    --platform=managed

# Chatbot App Service deploy Cloud Run-ra
echo "Chatbot App service deploy..."
gcloud run deploy $SERVICE_NAME_APP \
    --image ${ARTIFACT_REPO}/${APP_IMAGE_NAME} \
    --allow-unauthenticated \
    --cpu=2 --memory=8Gi \
    --port=8000 \
    --platform=managed \
    --set-env-vars OLLAMA_HOST="http://$SERVICE_NAME_OLLAMA:11434"

echo "Az összes szolgáltatás sikeresen telepítve. URL-ek lekérése..."

echo "Ollama Service URL:"
gcloud run services describe $SERVICE_NAME_OLLAMA --region=$REGION --format='value(status.url)'

echo "Neo4j Service URL:"
gcloud run services describe $SERVICE_NAME_NEO4J --region=$REGION --format='value(status.url)'

echo "Chatbot App Service URL:"
gcloud run services describe $SERVICE_NAME_APP --region=$REGION --format='value(status.url)'

