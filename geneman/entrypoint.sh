#!/usr/bin/env bash
set -euo pipefail

REPO=/app/GeneMAN
WEIGHTS=/weights
PM="$WEIGHTS/pretrained_models"     # poids -> sur le VOLUME (persistant)
EXT="$WEIGHTS/extern"
MARKER="$WEIGHTS/.geneman_weights_v2"

echo "==> GeneMAN entrypoint"

mkdir -p "$PM/seg" "$EXT"

# Symlinks repo -> volume, recréés À CHAQUE démarrage pour survivre à la
# recréation du conteneur (sinon /app/GeneMAN/pretrained_models, qui est DANS
# l'image, repart vide et on perd SAM/HumanNorm/Sapiens).
rm -rf "$REPO/pretrained_models"
ln -sfn "$PM" "$REPO/pretrained_models"
mkdir -p "$REPO/extern"
ln -sfn "$EXT/tets" "$REPO/extern/tets"

if [[ -f "$MARKER" ]]; then
  echo "==> Poids déjà présents (volume), skip du téléchargement."
else
  echo "==> Téléchargement des poids depuis HuggingFace (vers le volume)..."
  if [[ -n "${HF_TOKEN:-}" ]]; then
    huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential || true
  fi

  # 1) Poids GeneMAN (dataset wwt117/GeneMAN : pretrained_models + tets)
  huggingface-cli download wwt117/GeneMAN --repo-type dataset \
      --local-dir "$WEIGHTS/GeneMAN_hub"
  cp -r "$WEIGHTS/GeneMAN_hub/pretrained_models/." "$PM/"
  cp -r "$WEIGHTS/GeneMAN_hub/tets" "$EXT/" 2>/dev/null || true

  # 2) HumanNorm : repo HF (majuscule) -> dossier MINUSCULE attendu par les configs
  dl_hn() { huggingface-cli download "xanderhuang/$1" --local-dir "$PM/$2"; }
  dl_hn Normal-adapted-sd1.5     normal-adapted-sd1.5
  dl_hn Depth-adapted-sd1.5      depth-adapted-sd1.5
  dl_hn Normal-aligned-sd1.5     normal-aligned-sd1.5
  dl_hn controlnet-normal-sd1.5  controlnet-normal-sd1.5

  # 3) SAM ViT-H (chemin attendu par preprocessing.py : pretrained_models/seg/)
  if [[ ! -f "$PM/seg/sam_vit_h_4b8939.pth" ]]; then
    wget -q -O "$PM/seg/sam_vit_h_4b8939.pth" \
      https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
  fi

  touch "$MARKER"
  echo "==> Téléchargement terminé."
fi

echo "==> GeneMAN prêt — démarrage de l'API sur le port 8000"
cd /app
exec uvicorn api:app --host 0.0.0.0 --port 8000
