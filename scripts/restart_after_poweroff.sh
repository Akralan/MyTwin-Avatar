#!/usr/bin/env bash
# À lancer APRÈS un power off / on de l'instance.
# Le /scratch (éphémère) est vidé à l'arrêt : images + poids sont perdus.
# MAIS le déplacement Docker->scratch n'est PAS à refaire : /etc/docker/daemon.json
# vit sur le disque root et persiste. Il faut juste : vérifier le scratch,
# (re)appliquer la config par sécurité, rebuilder, re-télécharger.
set -euo pipefail

echo "==> 1. Vérification du /scratch"
if ! mountpoint -q /scratch; then
  echo "!! /scratch n'est PAS monté. Ne build pas (ça remplirait le disque root)."
  echo "   Repère le disque puis monte-le :"
  echo "     lsblk"
  echo "     sudo mkfs.ext4 /dev/sdb     # SEULEMENT s'il est vide/non formaté"
  echo "     sudo mount /dev/sdb /scratch"
  exit 1
fi
echo "   OK : $(df -h /scratch | tail -1)"

echo "==> 2. (sécurité) Docker pointe bien sur le scratch"
echo '{ "data-root": "/scratch/docker" }' | sudo tee /etc/docker/daemon.json >/dev/null
sudo mkdir -p /scratch/docker
sudo systemctl restart docker
docker info | grep "Docker Root Dir"

echo "==> 3. Rebuild des images + démarrage (images perdues -> ~10-15 min)"
git pull
docker compose up -d --build

echo "==> Terminé. Les poids se re-téléchargent (GeneMAN au boot, FLUX au 1er texturing)."
echo "    Suivi : docker compose logs -f"
