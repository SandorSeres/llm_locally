#!/bin/bash
# Exportáld a .env fájlban található változókat
set -a  # Automatikusan exportálja a változókat
source .env
set +a  # Kikapcsolja az automatikus exportálást
docker-compose -f ollama-compose.yaml build  #--no-cache
docker-compose -f ollama-compose.yaml up #--force-recreate

