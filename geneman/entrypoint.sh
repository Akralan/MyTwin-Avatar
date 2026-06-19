#!/usr/bin/env bash
set -euo pipefail

REPO=/app/GeneMAN
WEIGHTS=/weights
MARKER="$WEIGHTS/.geneman_weights_ok"

echo "==> GeneMAN entrypoint"

if [[ -f "$MARKER" ]]; then
  echo "==> Poids déjà présents, skip du téléchargement."
else
  echo "==> Téléchargement des poids depuis HuggingFace..."

  # Token HF (nécessaire pour certains modèles gated) — passé via .env
  if [[ -n "${HF_TOKEN:-}" ]]; then
    huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential || true
  fi

  mkdir -p "$REPO/pretrained_models" "$REPO/extern" "$REPO/pretrained_models/seg"

  # 1) Poids GeneMAN (dataset wwt117/GeneMAN : dossiers pretrained_models + tets)
  huggingface-cli download wwt117/GeneMAN --repo-type dataset \
      --local-dir "$WEIGHTS/GeneMAN_hub"
  cp -r "$WEIGHTS/GeneMAN_hub/pretrained_models/." "$REPO/pretrained_models/"
  cp -r "$WEIGHTS/GeneMAN_hub/tets" "$REPO/extern/" 2>/dev/null || true

  # 2) HumanNorm (modèles SD adaptés)
  for M in Normal-adapted-sd1.5 Depth-adapted-sd1.5 Normal-aligned-sd1.5 controlnet-normal-sd1.5; do
    huggingface-cli download "xanderhuang/$M" \
        --local-dir "$REPO/pretrained_models/$M"
  done

  # 3) SAM ViT-H (segmentation)
  if [[ ! -f "$REPO/pretrained_models/seg/sam_vit_h_4b8939.pth" ]]; then
    wget -q -O "$REPO/pretrained_models/seg/sam_vit_h_4b8939.pth" \
      https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
  fi

  touch "$MARKER"
  echo "==> Téléchargement terminé."
fi

echo "==> GeneMAN prêt — démarrage de l'API sur le port 8000"
cd /app
exec uvicorn api:app --host 0.0.0.0 --port 8000
