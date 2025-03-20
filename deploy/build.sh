#!/bin/bash
cd ..
#docker build -t sseres/ollama:latest -f Dockerfile.ollama .
docker build -t sseres/chatbot_app:latest -f Dockerfile .
#docker push sseres/ollama:latest
docker push sseres/chatbot_app:latest

