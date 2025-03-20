#!/bin/bash

# Ellenőrizzük, hogy az nvidia-container-runtime létezik-e
if ! command -v nvidia-container-runtime &> /dev/null
then
    echo "NVIDIA Container Toolkit nem található, telepítés elkezdése..."
    
    # NVIDIA repository konfigurálása
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
    && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list \
    && sudo apt-get update
    
    # NVIDIA Container Toolkit telepítése
    sudo apt-get install -y nvidia-container-toolkit

    # Runtime konfigurálása
    sudo nvidia-ctk runtime configure --runtime=docker

    # Docker újraindítása
    sudo systemctl restart docker

    echo "NVIDIA Container Toolkit telepítve és beállítva."
else
    echo "NVIDIA Container Toolkit már telepítve van."
fi

# Ellenőrizzük, hogy az NVIDIA runtime elérhető-e a Dockerben
if ! docker info | grep -q "Runtimes: nvidia"; then
    echo "NVIDIA runtime nem elérhető a Dockerben. Újrakonfigurálás..."
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker
fi

# Ellenőrizzük, hogy van-e futó NVIDIA konténer
if ! docker ps --format '{{.Image}}' | grep -q "nvidia"; then
    echo "Nincs futó NVIDIA konténer."
else
    echo "NVIDIA konténer jelenleg fut."
fi

