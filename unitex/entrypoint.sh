#!/usr/bin/env bash
set -euo pipefail

echo "==> UniTEX entrypoint"

# Token HF (au cas où certains checkpoints soient gated)
if [[ -n "${HF_TOKEN:-}" ]]; then
  huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential || true
fi

# UniTEX télécharge ses checkpoints automatiquement au premier appel du pipeline.
# Le cache HF est sur /weights (HF_HOME) -> persisté, pas de re-download.
echo "==> UniTEX prêt — démarrage de l'API sur le port 8000"
echo "    (les checkpoints se téléchargent au 1er texturing)"
cd /app/UniTEX
exec uvicorn api:app --host 0.0.0.0 --port 8000
