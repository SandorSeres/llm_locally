#!/bin/bash

# Paraméterek
PROJECT_ID="your-google-cloud-project-id"
SERVICE_NAME="demo-service"
IMAGE_NAME="demo-app"
REGION="us-central1"

# Cloud Run szolgáltatás eltávolítása
echo "Removing Cloud Run service..."
gcloud run services delete $SERVICE_NAME --region $REGION --quiet

# Docker image törlése a Container Registry-ből
echo "Removing Docker image from Container Registry..."
gcloud container images delete gcr.io/$PROJECT_ID/$IMAGE_NAME --quiet

echo "Cleanup completed."

