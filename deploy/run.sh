#!/bin/bash

cd /app
# Exportáld a .env fájlban található változókat
set -a  # Automatikusan exportálja a változókat
source ./.env
set +a  # Kikapcsolja az automatikus exportálást
python3 start.py
