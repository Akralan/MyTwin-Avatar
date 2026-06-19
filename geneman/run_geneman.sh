#!/usr/bin/env bash
# Reconstruction GeneMAN GÉOMÉTRIE pour une seule image.
# Dérivé de script/preprocess.sh + script/run.sh du repo (étapes 1-2 uniquement :
# la texture est déléguée à UniTEX).
#
#   run_geneman.sh <image_in> <out_dir>
# Produit : <out_dir>/model.obj (mesh géométrie) + <out_dir>/input_fg.png (image détourée)
set -euo pipefail

IMAGE="$1"
OUT="$2"
cd /app/GeneMAN

WORK="$(mktemp -d)"
mkdir -p "$WORK/raw"
cp "$IMAGE" "$WORK/raw/input.png"
DATA="$WORK/processed"
EXP="$WORK/outputs"
ID="input"
# Timestamp FIXE et NON VIDE : si vide, threestudio le traite comme None et génère
# un horodatage différent à chaque appel (train vs export) -> last.ckpt introuvable.
TS="_run"

# Helpers : localisent le fichier le plus récent (robuste au nom exact du dossier).
find_ckpt() { find "$1" -name last.ckpt -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-; }
find_obj()  { find "$1" -name '*.obj'   -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-; }

echo "==> [1/3] Préprocessing (détourage, normales, keypoints, caption)"
python preprocessing.py "$WORK/raw" --output_path "$DATA" --recenter --enable_captioning

IMG_FG="$DATA/${ID}_fg.png"
NORMAL="$DATA/${ID}_normal.png"
KPTS="$DATA/${ID}_landmarks.npy"
PROMPT="$(cut -d'|' -f1 < "$DATA/${ID}_caption.txt")"
echo "    prompt: $PROMPT"

echo "==> [2/3] Stage 1 — initialisation géométrie (NeRF)"
python launch.py --config configs/geneman-geometry-init.yaml --train \
    tag="$ID" timestamp="$TS" exp_root_dir="$EXP" \
    data.image_path="$IMG_FG" system.prompt_processor.prompt="$PROMPT"

INIT_CKPT="$(find_ckpt "$EXP/geneman-geometry-init")"
[[ -n "$INIT_CKPT" ]] || { echo "!! Stage 1 : aucun last.ckpt trouvé"; exit 1; }
python launch.py --config configs/geneman-geometry-init.yaml --export \
    tag="$ID" timestamp="$TS" exp_root_dir="$EXP" resume="$INIT_CKPT" \
    data.image_path="$IMG_FG" system.prompt_processor.prompt="$PROMPT" \
    system.exporter_type=mesh-exporter \
    system.exporter.save_texture=False system.exporter.save_uv=False \
    system.geometry.isosurface_method=mc-cpu \
    system.geometry.isosurface_resolution=256

MESH_INIT="$(find_obj "$EXP/geneman-geometry-init")"
[[ -n "$MESH_INIT" ]] || { echo "!! Stage 1 : aucun mesh exporté"; exit 1; }

echo "==> [3/3] Stage 2 — sculpting géométrie"
python launch.py --config configs/geneman-geometry-sculpt.yaml --train \
    tag="$ID" timestamp="$TS" exp_root_dir="$EXP" data.sampling_type="full_body" \
    data.image_path="$IMG_FG" data.normal_path="$NORMAL" data.keypoints_path="$KPTS" \
    system.prompt_processor.prompt="$PROMPT, black background, normal map" \
    system.prompt_processor_add.prompt="$PROMPT, black background, depth map" \
    system.prompt_processor.human_part_prompt=false \
    system.geometry.shape_init="mesh:$MESH_INIT"

SCULPT_CKPT="$(find_ckpt "$EXP/geneman-geometry-sculpt")"
[[ -n "$SCULPT_CKPT" ]] || { echo "!! Stage 2 : aucun last.ckpt trouvé"; exit 1; }
python launch.py --config configs/geneman-geometry-sculpt.yaml --export \
    tag="$ID" timestamp="$TS" exp_root_dir="$EXP" resume="$SCULPT_CKPT" \
    data.image_path="$IMG_FG" system.prompt_processor.prompt="$PROMPT" \
    system.exporter_type=mesh-exporter system.exporter.save_texture=False \
    system.exporter.fmt=obj

echo "==> Récupération du mesh final"
mkdir -p "$OUT"
FINAL="$(find_obj "$EXP/geneman-geometry-sculpt")"
[[ -n "$FINAL" ]] || { echo "!! aucun mesh final trouvé"; exit 1; }
cp "$FINAL" "$OUT/model.obj"
cp "$IMG_FG" "$OUT/input_fg.png"   # image détourée -> à renvoyer ensuite à UniTEX
echo "==> Terminé : $OUT/model.obj"
