#!/usr/bin/env bash
# Bootstrap d'une instance GPU Scaleway fraîche.
# À lancer UNE fois après connexion SSH à l'instance.
#
#   curl -fsSL https://raw.githubusercontent.com/CHANGE_ME/mytwin-avatar-poc/main/scripts/bootstrap_instance.sh | bash
#   (ou : git clone ... && bash scripts/bootstrap_instance.sh)
set -euo pipefail

echo "==> Vérification du GPU NVIDIA"
nvidia-smi || { echo "!! Driver NVIDIA absent. Choisis une image Scaleway GPU avec drivers."; exit 1; }

# --- Docker ---
if ! command -v docker &>/dev/null; then
  echo "==> Installation de Docker"
  curl -fsSL https://get.docker.com | sh
fi

# --- NVIDIA Container Toolkit (pour --gpus / device nvidia) ---
if ! docker info 2>/dev/null | grep -qi nvidia; then
  echo "==> Installation du NVIDIA Container Toolkit"
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
    sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
  sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker
fi

echo "==> Test GPU dans un conteneur"
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# --- Login GHCR (images privées) + déploiement ---
if [[ ! -f .env ]]; then
  echo "==> Crée d'abord .env  (cp .env.example .env  puis renseigne HF_TOKEN et REGISTRY)"
  exit 1
fi

echo "==> Récupération et démarrage des modèles"
docker compose pull
docker compose up -d

echo "==> Terminé. Suivi des logs :  docker compose logs -f"
