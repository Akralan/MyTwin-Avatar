#!/usr/bin/env bash
# Reconstruction GeneMAN GÉOMÉTRIE pour une seule image.
# Dérivé de script/preprocess.sh + script/run.sh du repo (étapes 1-2 uniquement :
# la texture est déléguée à UniTEX).
#
#   run_geneman.sh <image_in> <out_dir>
# Produit : <out_dir>/model.obj (mesh géométrie) + <out_dir>/input_fg.png (image détourée)
#
# ⚠️ Les chemins exacts d'export (it5000-export, etc.) sont issus de run.sh du repo ;
#    à confirmer au 1er run réel. Surchargeable globalement via GENEMAN_CMD dans api.py.
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
TS=""

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

INIT_CKPT="$EXP/geneman-geometry-init/${ID}${TS}/ckpts/last.ckpt"
python launch.py --config configs/geneman-geometry-init.yaml --export \
    tag="$ID" timestamp="$TS" exp_root_dir="$EXP" resume="$INIT_CKPT" \
    data.image_path="$IMG_FG" system.prompt_processor.prompt="$PROMPT" \
    system.exporter_type=mesh-exporter \
    system.exporter.save_texture=False system.exporter.save_uv=False \
    system.geometry.isosurface_method=mc-cpu \
    system.geometry.isosurface_resolution=256

MESH_INIT="$EXP/geneman-geometry-init/${ID}${TS}/save/it5000-export/model.obj"

echo "==> [3/3] Stage 2 — sculpting géométrie"
python launch.py --config configs/geneman-geometry-sculpt.yaml --train \
    tag="$ID" timestamp="$TS" exp_root_dir="$EXP" data.sampling_type="full_body" \
    data.image_path="$IMG_FG" data.normal_path="$NORMAL" data.keypoints_path="$KPTS" \
    system.prompt_processor.prompt="$PROMPT, black background, normal map" \
    system.prompt_processor_add.prompt="$PROMPT, black background, depth map" \
    system.prompt_processor.human_part_prompt=false \
    system.geometry.shape_init="mesh:$MESH_INIT"

SCULPT_CKPT="$EXP/geneman-geometry-sculpt/${ID}${TS}/ckpts/last.ckpt"
python launch.py --config configs/geneman-geometry-sculpt.yaml --export \
    tag="$ID" timestamp="$TS" exp_root_dir="$EXP" resume="$SCULPT_CKPT" \
    data.image_path="$IMG_FG" system.prompt_processor.prompt="$PROMPT" \
    system.exporter_type=mesh-exporter system.exporter.save_texture=False \
    system.exporter.fmt=obj

echo "==> Récupération du mesh final"
mkdir -p "$OUT"
# Le dernier .obj exporté par le stage sculpt = mesh géométrie final
FINAL="$(find "$EXP/geneman-geometry-sculpt" -name '*.obj' -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)"
cp "$FINAL" "$OUT/model.obj"
cp "$IMG_FG" "$OUT/input_fg.png"   # image détourée -> à renvoyer ensuite à UniTEX
echo "==> Terminé : $OUT/model.obj"
