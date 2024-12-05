#!/bin/bash

gcloud run deploy neo4j-service \
    --image gcr.io/$PROJECT_ID/neo4j-service \
    --region us-central1 \
    --platform managed \
    --allow-unauthenticated \
    --cpu 1 \
    --memory 2Gi \
    --port 7474


gcloud run services describe neo4j-service --region us-central1 --format="value(status.url)"

gcloud run deploy fastapi-app \
    --image gcr.io/$PROJECT_ID/fastapi-app \
    --region us-central1 \
    --platform managed \
    --allow-unauthenticated \
    --cpu 1 \
    --memory 1Gi \
    --port 8000 \
    --set-env-vars "NEO4J_URL=https://neo4j-service-xxxxxxxxxx-uc.a.run.app,NEO4J_USERNAME=neo4j,NEO4J_PASSWORD=yourpassword"


echo "Deployment completed."

