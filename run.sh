#!/bin/bash
# Exportáld a .env fájlban található változókat
set -a  # Automatikusan exportálja a változókat
source .env
set +a  # Kikapcsolja az automatikus exportálást
docker-compose -f ollama-compose.yaml up --build --force-recreate

