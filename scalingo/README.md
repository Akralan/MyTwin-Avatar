# MyTwin Avatar

Photo → avatar 3D. **Deux services découplés**, backend sans état, avatars
stockés sur l'appareil de l'utilisateur (aucune base de données).

```
frontend/  ── Scalingo   : sert la page + assets client ; le navigateur orchestre
backend/   ── Scaleway   : conteneur serverless, 3 endpoints stateless (compute)
```

Le navigateur est le chef d'orchestre : il lance la génération du corps (Meshy,
proxifiée par le backend), capture le visage (MediaPipe FaceLandmarker en WASM),
demande la greffe au backend, puis **stocke l'avatar dans IndexedDB** (galerie).
Aucun état côté serveur : pas de DB, pas de disque persistant, pas de thread.

## Backend (`backend/`) — Scaleway Serverless Container

Endpoints (sans état) :

| Méthode | Route | Rôle |
|---|---|---|
| `POST` | `/body` | détourage (rembg) + création tâche Meshy → `{task_id}` |
| `GET`  | `/body/status?task_id=` | proxy court vers Meshy → `{status, progress}` |
| `POST` | `/graft` | `face.glb` + `task_id` → pipeline CPU → renvoie `avatar.glb` |
| `GET`  | `/healthz` | état |

Le pipeline de greffe (`pipeline/`) est **Python CPU pur** (numpy/scipy/trimesh/
opencv, rasteriseur logiciel — pas de GPU, pas de Blender). `libgl1`/`libglib2.0-0`
requis au runtime (opencv). La greffe est synchrone (~1 min CPU).

Variables : `MESHY_API_KEY`, `MESHY_*`, `REMOVE_BG`, `REMBG_MODEL`, `CORS_ORIGIN`
(origine du frontend, défaut `*` pour la démo), `USE_LOCAL_BODY`/`USE_LOCAL_FACE`
(modes test : greffent `corps.glb`/`visage.glb` sans appeler Meshy).

Build & run (Docker) :

```bash
cd backend
docker build -t mytwin-api .
docker run -p 8080:8080 -e MESHY_API_KEY=... -e CORS_ORIGIN=https://<frontend> mytwin-api
```

Déploiement : push l'image sur Scaleway Container Registry → Serverless Container
(scale-to-zero ; timeout requête ≥ durée de greffe ; ~2 Go de RAM).

## Frontend (`frontend/`) — Scalingo

Flask minimal (Flask + gunicorn) : sert `index.html`, `models/face_landmarker.task`
(asset client) et injecte l'URL du backend via `API_BASE`. Aucun secret, aucun
compute. La galerie et le parcours sont 100 % côté navigateur.

Variable : `API_BASE` = URL du backend Scaleway.

## Développement local (2 process)

```bash
# 1) backend (modes locaux : aucun appel/crédit Meshy)
cd backend && USE_LOCAL_BODY=1 USE_LOCAL_FACE=1 PORT=5001 python api.py

# 2) frontend (pointe sur le backend local)
cd frontend && API_BASE=http://localhost:5001 PORT=8000 python app.py
```

Ouvre http://127.0.0.1:8000 (la caméra exige un contexte sécurisé : `127.0.0.1`
est accepté ; pour un test mobile via IP réseau, servir en HTTPS).

Modes test : `USE_LOCAL_BODY=1` greffe `corps.glb` (pas de Meshy) ;
`USE_LOCAL_FACE=1` greffe `visage.glb` (ignore le visage capturé).

## Notes démo

- **Poids des avatars** : un avatar fait ~60 Mo (corps haute résolution + textures).
  IndexedDB tient pour quelques avatars ; peut être purgé par le navigateur sous
  pression de stockage (surtout iOS). D'où le bouton **Télécharger** (copie durable).
  Levier d'allègement : Meshy `should_remesh` + texture réduite.
- **Protection crédits Meshy** : volontairement absente (démo). À rajouter (BFF /
  gate) avant une mise en prod ouverte.
- **Cold start** : la 1re requête après inactivité charge rembg+mediapipe
  (~10-30 s). Mettre `min-scale=1` pendant une démo live si gênant.
