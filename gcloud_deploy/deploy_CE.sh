#!/usr/bin/env bash

# Beállítások
PROJECT_ID="sas-ollama"                   # A saját Google Cloud projekted ID-ja
ZONE="us-central1-a"                      # Régió, ahol a VM-et létrehozod
INSTANCE_NAME="ollama-gpu-vm"             # A VM neve
MACHINE_TYPE="custom-4-16384"             # 4 CPU, 16 GiB RAM
GPU_TYPE="nvidia-tesla-t4"                # A használt GPU típusa
GPU_COUNT=1                               # GPU-k száma
IMAGE_FAMILY="ubuntu-2004-lts"            # OS képfájl
IMAGE_PROJECT="ubuntu-os-cloud"           # Kép projektje
DISK_SIZE="100GB"                         # Boot disk mérete

# Projekt és régió beállítása
gcloud config set project $PROJECT_ID
gcloud config set compute/zone $ZONE

# Compute Engine VM létrehozása GPU-val
echo ">>> VM létrehozása GPU támogatással..."
gcloud compute instances create $INSTANCE_NAME \
    --machine-type $MACHINE_TYPE \
    --accelerator type=$GPU_TYPE,count=$GPU_COUNT \
    --image-family $IMAGE_FAMILY \
    --image-project $IMAGE_PROJECT \
    --boot-disk-size $DISK_SIZE \
    --maintenance-policy TERMINATE \
    --restart-on-failure

# NVIDIA GPU driver telepítése
echo ">>> NVIDIA driver telepítése a VM-en..."
gcloud compute ssh $INSTANCE_NAME --command="
    sudo apt update &&
    sudo apt install -y nvidia-driver-470
"

# Docker és NVIDIA Docker telepítése
echo ">>> Docker és NVIDIA Docker telepítése..."
gcloud compute ssh $INSTANCE_NAME --command="
    sudo apt install -y docker.io &&
    sudo usermod -aG docker \$USER &&
    curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - &&
    curl -s -L https://nvidia.github.io/nvidia-docker/ubuntu18.04/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list &&
    sudo apt update &&
    sudo apt install -y nvidia-docker2 &&
    sudo systemctl restart docker
"

# Ollama Docker image futtatása
echo ">>> Ollama Docker konténer indítása..."
gcloud compute ssh $INSTANCE_NAME --command="
    sudo docker run --gpus all -p 8080:8080 ollama/ollama:latest
"

# VM IP-címének lekérdezése
echo ">>> A VM IP-címe:"
gcloud compute instances describe $INSTANCE_NAME --format='get(networkInterfaces[0].accessConfigs[0].natIP)'

echo ">>> Deploy sikeres! Az Ollama szolgáltatás a http://<VM_PUBLIC_IP>:8080 címen érhető el."
    
