# Plan — Détails du visage (teinte) + stries des joues + reco perf

Date : 2026-07-24 · État du repo : HEAD `5dcf932`

## Contexte

Le pipeline de greffe (`backend/pipeline/`, orchestré par `replace_face` dans `orchestrator.py:137`) fonctionne mais présente deux défauts visuels :

1. **Perte de détails sur certaines photos** : `match_skin_tone` (`backend/pipeline/texture_blender.py:49`) fait un transfert Reinhard en Lab. Le gain multiplicatif `ratio = clip(body_std/face_std, 0.6, 1.6)` s'applique aussi au canal L : quand le selfie est plus contrasté que la peau du corps Meshy (lumière dure), gain < 1 → jusqu'à −40 % de contraste de luminance sur toute la texture → barbe, ombres du nez, pores écrasés.
2. **Stries sur les joues** : selfie frontal → triangles des joues inclinés vers l'oreille couverts par une bande étroite de pixels (étirement 1/cos θ). Le mécanisme prévu (`stretch_fade` dans `add_matched_grain`, `texture_grain.py:259`) a été désactivé au commit `afea3ff` (`orchestrator.py:221-223`) parce qu'il supprimait TOUTE la haute fréquence des joues (perte de vrai détail), pas seulement les stries.

Décision : corriger P1 + P2 ; les améliorations perf (P3) sont une **liste de recommandations seulement**, rien à implémenter pour l'instant.

Assets de test versionnés : `backend/corps.glb` + `backend/visage.glb`. Test local sur le port **8000** (jamais 5000).

---

## Phase 0 — Harnais de test hors serveur

**Nouveau fichier `backend/tools/test_pipeline.py`** (CLI, hors Flask), 3 sous-commandes :

- `tone` : charge visage.glb + corps.glb (`pipeline.loader`, `head_locator.locate_head`), exécute `match_skin_tone` ancien vs nouveau, sauve les atlas avant/après en PNG + métrique de détail HF (std du Laplacien du canal L sur le masque peau). Quelques secondes.
- `stretchmask` : calcule `stretch_mask_vertex` (`texture_grain.py:199`), rasterise via `texture_blender._rasterize_uv` et sauve une heatmap superposée à l'atlas — pour calibrer `stretch_band` visuellement.
- `full` : appelle `replace_face(corps.glb, visage.glb, out)` avec kwargs CLI (`--side-trim-after`, `--stretch-band`, `--stretch-mode`, `--debug-dir`) + timings.

**Instrumentation dans `replace_face`** : dict `timings` par étape ajouté à `PipelineReport` (`orchestrator.py:21`), loggé dans `api.py` ; paramètre `debug_dir: str | None = None` qui sauve l'atlas après chaque étape texture (`01_tone.png` … `06_gutter.png`). Défauts = comportement inchangé.

Sorties dans `backend/tools/out/` (gitignoré).

## Phase 1 — P1 : préserver la luminance dans `match_skin_tone`

**Fichier : `backend/pipeline/texture_blender.py:49-82`.**

Correctif : borner le gain du canal L pour ne **jamais comprimer** le contraste. Un gain L ≥ 1 ne peut pas détruire de détail ; le shift de moyenne L (vraie carnation) est conservé ; le résidu spatial basse fréquence est déjà absorbé en aval par `harmonic_tone_match` (:191) et `feather_border` (:154). Chroma (a, b) : Reinhard inchangé.

```python
def match_skin_tone(face, body, hf, std_clip=(0.6, 1.6), strength: float = 1.0,
                    luma_gain_clip: tuple = (1.0, 1.4)) -> trimesh.Trimesh:
    ...
    gain = strength * ratio + (1 - strength)
    # L : ne jamais comprimer (gain<1 écrase la HF de luminance : barbe, ombres, pores).
    gain[0] = float(np.clip(gain[0], *luma_gain_clip))
```

Borne haute 1.4 : si le selfie est plus plat que le corps on amplifie modérément (pas 1.6, pour ne pas amplifier le bruit). Ancien comportement récupérable via `luma_gain_clip=(0.6, 1.6)`. Aucun changement d'appel dans l'orchestrateur.

**Test** : `python backend/tools/test_pipeline.py tone` — le métrique Laplacien-L doit rester ≥ ~95 % de l'original quand `body_std_L < face_std_L` ; comparer les PNG ; puis `full` pour vérifier la couture.

## Phase 2 — P2 : stries des joues → dé-streak DIRECTIONNEL

Idée clé : les stries sont **anisotropes** (HF pleine selon v, déjà étalée selon u par la projection frontale). Au lieu de retirer toute la HF (l'ancien mode isotrope qui a été désactivé), on ne lisse la HF que **selon v** dans les zones étirées : les stries disparaissent, le vrai détail (déjà lissé en u par l'étirement lui-même + basse/moyenne fréquence) est conservé.

### 2.1 `backend/pipeline/texture_grain.py` — `add_matched_grain` (:259)

Nouveaux kwargs (défauts rétro-compatibles) :

```python
stretch_mode: str = "isotropic",     # "isotropic" | "directional"
streak_sigma: float | None = None,   # sigma du lissage v (px upsampled)
```

Remplacement du bloc `if supp is not None:` (:355-364) :

```python
if supp is not None:
    import cv2
    ds = detail_sigma if detail_sigma is not None else max(2.0, 2.0 * cell_px)
    low = cv2.GaussianBlur(arr, (0, 0), sigmaX=ds, sigmaY=ds)
    detail = arr - low
    w = supp * gate                      # gate luminance inchangé (sourcils/yeux)
    if stretch_mode == "directional":
        ss = streak_sigma if streak_sigma is not None else max(2.5, 2.5 * cell_px)
        k = cv2.getGaussianKernel(int(6 * ss) | 1, ss)
        det_v = cv2.sepFilter2D(detail, -1, np.array([1.0], np.float32), k)  # flou 1-D vertical
        out = low + (1.0 - w) * detail + w * det_v + grain * gate
    else:
        out = low + (1.0 - w) * detail + grain * gate      # legacy isotrope
else:
    out = arr + grain * gate
```

### 2.2 `backend/pipeline/orchestrator.py` — réactivation (:221-223)

```python
stitched = texture_grain.add_matched_grain(stitched, body, hf,
                                           wavelength_scale=1.0,
                                           stretch_fade=True,
                                           stretch_mode="directional",
                                           stretch_band=(2.0, 3.5),
                                           stretch_strength=1.0)
```

Bande (2.0, 3.5) — plus tardive que l'originale (1.8, 3.2) car le mode directionnel est moins destructif. Calibration : stries persistantes près de la couture → lo=1.8 ; détail joue qui disparaît → (2.3, 4.0). Mettre à jour le commentaire :218-220.

### 2.3 Option géométrique (test seulement, pas de changement de défaut)

- `full --side-trim-after 1` (déjà plumbé jusqu'à `junction.trim_sides`) : les joues les plus inclinées restent au corps Meshy.
- Si retenu : tester `inflate_side` 0.93-0.94 (`orchestrator.py:124` → `mesh_surgery.cut_to_silhouette`) pour que le trou suive le masque rétréci. À valider en 3D (risque de découvrir la peau du corps).

**Ne PAS intégrer `fix_cheek_stretch.py`** (racine scalingo/) : re-étalement UV codé en dur pour textures 480×640, interagit avec toutes les étapes aval qui rasterisent les UV. Piste long terme seulement.

**Tests** : (1) `stretchmask` pour visualiser où (2.0, 3.5) mord ; (2) `full --debug-dir` en 3 modes (off / isotropic / directional), comparer `04_grain.png` ; (3) contrôle visuel GLB dans le frontend local port 8000 (`local_body=1&local_face=1`).

## Phase 3 — P3 : recommandations perf/robustesse (liste, rien à coder pour l'instant)

**Quick wins** (aucun changement visuel) :
1. Mutualiser le NlMeans du corps : `measure_body_grain` (`texture_grain.py:46`) et `measure_body_grain_wavelength` (:123) font chacun un `fastNlMeansDenoisingColored` (le premier sur l'atlas 2K entier — c'est le plus gros poste CPU du pipeline texture). Un helper commun croppé au bbox peau → ÷2 ou mieux.
2. Réutiliser le raster de couverture UV : `_rasterize_uv` (boucle Python par triangle, à 3072²) tourne 3× avec les mêmes UV (grain, fringe, gutter) → calculer une fois et passer le résultat.
3. Cacher le FaceLandmarker MediaPipe (`landmark_detector.py:36-52` recrée le modèle .task à chaque appel ; `_refine_head_frame` détecte jusqu'à 3×) → singleton module, 1-3 s gagnées.
4. Timings par étape (fait en Phase 0) — prérequis pour mesurer le reste sur Scaleway vs local.

**Gains moyens** :
5. Mesure du grain corps dans un thread lancé juste après `load_body`, `join()` avant l'étape grain (recouvre le NlMeans avec les ~30-60 s de géométrie/landmarks).
6. `upsample=3 → 2` dans `add_matched_grain` : ~2.25× moins de pixels pour toutes les étapes aval (fringe, gutter, export, mémoire client) — change légèrement le rendu du grain, à valider visuellement.
7. Export : ré-encoder l'atlas final en JPEG q92 plutôt que PNG 3072² (temps d'export + taille de la réponse /graft).
8. Robustesse `_skin_mask` (seuils `r>=g>=b, lum>60`) : fallback percentile si le masque < 5 % des candidats (peaux foncées / balance des blancs froide → stats polluées par cheveux/fond).
9. Garde de concurrence `/graft` (1 worker × 8 threads gunicorn) : `BoundedSemaphore(1)` non bloquant → 429 « occupé » si une greffe est déjà en cours (2 greffes simultanées se partagent le CPU et peuvent dépasser les 900 s).

**À ne pas faire** : monter la résolution du rendu landmarks (600px suffit, la précision vient du position buffer interpolé), rendre /graft asynchrone, intégrer fix_cheek_stretch.py.

---

## Ordre d'exécution & validation

1. Phase 0 → baseline `full` sur corps.glb/visage.glb (timings + PNG de référence AVANT tout changement).
2. Phase 1 (~10 lignes) → test `tone`, puis `full`, contrôle visuel port 8000.
3. Phase 2 → `stretchmask` pour calibrer, `full --debug-dir` en 3 modes, matrice `side_trim_after ∈ {0,1}`, contrôle 3D.
4. Un commit par phase. P1 et P2 indépendants et réversibles par kwargs (`luma_gain_clip=(0.6,1.6)`, `stretch_fade=False`).

## Fichiers touchés

- `backend/pipeline/texture_blender.py` — P1 (`match_skin_tone`)
- `backend/pipeline/texture_grain.py` — P2 (`add_matched_grain`)
- `backend/pipeline/orchestrator.py` — réactivation stretch_fade + timings + debug_dir
- `backend/tools/test_pipeline.py` — nouveau harnais de test
- `backend/api.py` — log des timings (1 ligne)
