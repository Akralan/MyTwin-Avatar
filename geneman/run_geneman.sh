#!/usr/bin/env bash
# Reconstruction GeneMAN GÉOMÉTRIE pour une seule image.
# Dérivé de script/preprocess.sh + script/run.sh du repo (étapes 1-2 uniquement :
# la texture est déléguée à UniTEX).
#
#   run_geneman.sh <image_in> <out_dir>
# Produit : <out_dir>/model.obj (mesh géométrie) + <out_dir>/input_fg.png (image détourée)
#
# CACHE : le dossier de travail est indexé par le hash de l'image (sur le volume).
#   -> même image = on saute préprocessing/Stage 1/Stage 2 déjà faits.
#   Pour forcer un run propre : rm -rf /weights/geneman_cache/<hash>  (ou tout le dossier).
set -euo pipefail

IMAGE="$1"
OUT="$2"
cd /app/GeneMAN

ID="input"
# Timestamp FIXE et NON VIDE : si vide, threestudio le traite comme None et génère
# un horodatage différent à chaque appel (train vs export) -> last.ckpt introuvable.
TS="_run"

# Dossier de travail PERSISTANT, indexé par le contenu de l'image -> cache réutilisable.
HASH="$(md5sum "$IMAGE" | awk '{print $1}')"
WORK="/weights/geneman_cache/$HASH"
DATA="$WORK/processed"
EXP="$WORK/outputs"
mkdir -p "$WORK/raw"
cp -f "$IMAGE" "$WORK/raw/input.png"

# Helpers : localisent le fichier le plus récent (robuste au nom exact du dossier).
find_ckpt() { find "$1" -name last.ckpt -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-; }
find_obj()  { find "$1" -name '*.obj'   -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-; }

# ---------------------------------------------------------------- Préprocessing
if [[ -f "$DATA/${ID}_caption.txt" ]]; then
  echo "==> [1/3] Préprocessing déjà en cache, skip."
else
  echo "==> [1/3] Préprocessing (détourage, normales, keypoints, caption)"
  python preprocessing.py "$WORK/raw" --output_path "$DATA" --recenter --enable_captioning
fi

IMG_FG="$DATA/${ID}_fg.png"
NORMAL="$DATA/${ID}_normal.png"
KPTS="$DATA/${ID}_landmarks.npy"
# tr ',' ' ' : OmegaConf.from_cli interprète les virgules comme des séparateurs de
# liste -> une virgule dans le prompt casse l'affectation. On les retire.
PROMPT="$(cut -d'|' -f1 < "$DATA/${ID}_caption.txt" | tr ',' ' ')"
echo "    prompt: $PROMPT"

# ----------------------------------------------------------- Stage 1 (géométrie)
if [[ -n "$(find_obj "$EXP/geneman-geometry-init")" ]]; then
  echo "==> [2/3] Stage 1 déjà fait (mesh init en cache), skip."
else
  echo "==> [2/3] Stage 1 — initialisation géométrie (NeRF)"
  if [[ -z "$(find_ckpt "$EXP/geneman-geometry-init")" ]]; then
    # Le `|| echo` tolère le crash de la phase de TEST post-entraînement (rendu
    # d'aperçus, IndexError image vide) : le last.ckpt est déjà sauvegardé.
    python launch.py --config configs/geneman-geometry-init.yaml --train \
        tag="$ID" timestamp="$TS" exp_root_dir="$EXP" \
        data.image_path="$IMG_FG" system.prompt_processor.prompt="$PROMPT" \
        || echo "[warn] phase test post-entraînement Stage 1 échouée (non bloquant)"
  fi
  INIT_CKPT="$(find_ckpt "$EXP/geneman-geometry-init")"
  [[ -n "$INIT_CKPT" ]] || { echo "!! Stage 1 : aucun last.ckpt trouvé"; exit 1; }
  python launch.py --config configs/geneman-geometry-init.yaml --export \
      tag="$ID" timestamp="$TS" exp_root_dir="$EXP" resume="$INIT_CKPT" \
      data.image_path="$IMG_FG" system.prompt_processor.prompt="$PROMPT" \
      system.exporter_type=mesh-exporter \
      system.exporter.save_texture=False system.exporter.save_uv=False \
      system.geometry.isosurface_method=mc-cpu \
      system.geometry.isosurface_resolution=256
fi

MESH_INIT="$(find_obj "$EXP/geneman-geometry-init")"
[[ -n "$MESH_INIT" ]] || { echo "!! Stage 1 : aucun mesh exporté"; exit 1; }

# ------------------------------------------------------------- Stage 2 (sculpt)
if [[ -n "$(find_obj "$EXP/geneman-geometry-sculpt")" ]]; then
  echo "==> [3/3] Stage 2 déjà fait (mesh final en cache), skip."
else
  echo "==> [3/3] Stage 2 — sculpting géométrie"
  if [[ -z "$(find_ckpt "$EXP/geneman-geometry-sculpt")" ]]; then
    python launch.py --config configs/geneman-geometry-sculpt.yaml --train \
        tag="$ID" timestamp="$TS" exp_root_dir="$EXP" data.sampling_type="full_body" \
        data.image_path="$IMG_FG" data.normal_path="$NORMAL" data.keypoints_path="$KPTS" \
        system.prompt_processor.prompt="$PROMPT black background normal map" \
        system.prompt_processor_add.prompt="$PROMPT black background depth map" \
        system.prompt_processor.human_part_prompt=false \
        system.geometry.shape_init="mesh:$MESH_INIT" \
        || echo "[warn] phase test post-entraînement Stage 2 échouée (non bloquant)"
  fi
  SCULPT_CKPT="$(find_ckpt "$EXP/geneman-geometry-sculpt")"
  [[ -n "$SCULPT_CKPT" ]] || { echo "!! Stage 2 : aucun last.ckpt trouvé"; exit 1; }
  python launch.py --config configs/geneman-geometry-sculpt.yaml --export \
      tag="$ID" timestamp="$TS" exp_root_dir="$EXP" resume="$SCULPT_CKPT" \
      data.image_path="$IMG_FG" system.prompt_processor.prompt="$PROMPT" \
      system.exporter_type=mesh-exporter system.exporter.save_texture=False \
      system.exporter.fmt=obj
fi

echo "==> Récupération du mesh final"
mkdir -p "$OUT"
FINAL="$(find_obj "$EXP/geneman-geometry-sculpt")"
[[ -n "$FINAL" ]] || { echo "!! aucun mesh final trouvé"; exit 1; }
cp "$FINAL" "$OUT/model.obj"
cp "$IMG_FG" "$OUT/input_fg.png"   # image détourée -> à renvoyer ensuite à UniTEX
echo "==> Terminé : $OUT/model.obj"
