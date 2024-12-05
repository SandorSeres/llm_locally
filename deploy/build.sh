#!/bin/bash

PROJECT_ID="your-google-cloud-project-id"
docker build -t gcr.io/$PROJECT_ID/neo4j-service -f neo4j.Dockerfile .
docker push gcr.io/$PROJECT_ID/neo4j-service

docker build -t gcr.io/$PROJECT_ID/fastapi-app .
docker push gcr.io/$PROJECT_ID/fastapi-app


