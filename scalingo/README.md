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
(origine du frontend, défaut `*` pour la démo). Les modes test (greffer
`corps.glb`/`visage.glb` sans appeler Meshy) sont pilotés **par requête** via les
champs `local_body`/`local_face` — exposés en toggles dans les réglages du
frontend (désactivés par défaut), plus de variable d'env.

Build & run local (Docker) :

```bash
cd backend
docker build -t mytwin-api .
docker run -p 8080:8080 -e MESHY_API_KEY=... -e CORS_ORIGIN=https://<frontend> mytwin-api
```

Déploiement sur Scaleway (Container Registry → Serverless Container). Remplacer
`<namespace>` par le nom du namespace du registre ; région `fr-par` ici.

```bash
# 0) (une fois) créer un namespace de registre et récupérer une clé API Scaleway.
#    Login Docker au registre (user = "nologin", password = clé secrète Scaleway).
echo "$SCW_SECRET_KEY" | docker login rg.fr-par.scw.cloud -u nologin --password-stdin

# 1) Build en amd64 (Scaleway tourne en x86_64 ; --platform indispensable si build
#    depuis un Mac ARM, inoffensif sinon). Tag = chemin complet dans le registre.
cd backend
docker build --platform linux/amd64 -t rg.fr-par.scw.cloud/<namespace>/mytwin-api:latest .

# 2) Push de l'image
docker push rg.fr-par.scw.cloud/<namespace>/mytwin-api:latest

# 3) Pointer le Serverless Container sur la nouvelle image (ou via la console).
#    Régler les variables (MESHY_API_KEY, CORS_ORIGIN=https://<frontend>, …),
#    port 8080, ~2 Go de RAM, timeout requête >= durée de greffe (~300 s).
scw container container update <container-id> \
    registry-image=rg.fr-par.scw.cloud/<namespace>/mytwin-api:latest
```

Scale-to-zero : la 1re requête après inactivité paie le cold start (chargement
rembg + mediapipe). Le tag `:latest` étant mutable, un `docker push` suivi d'un
redéploiement du conteneur suffit pour livrer une nouvelle version.

## Frontend (`frontend/`) — Scalingo

Flask minimal (Flask + gunicorn) : sert `index.html`, `models/face_landmarker.task`
(asset client) et injecte l'URL du backend via `API_BASE`. Aucun secret, aucun
compute. La galerie et le parcours sont 100 % côté navigateur.

Variable : `API_BASE` = URL du backend Scaleway.

Déploiement Scalingo (monorepo : seul `frontend/` est poussé, remis à la racine
via `git subtree`, donc pas besoin de `PROJECT_DIR`) :

```bash
# depuis la racine du repo, sur la branche main
git subtree push --prefix scalingo/frontend scalingo master
```

`scalingo` = remote `git@ssh.osc-fr1.scalingo.com:demo-mytwin-avatar.git`.
Configurer `API_BASE` sur l'app (`scalingo --app demo-mytwin-avatar env-set
API_BASE=…`) avant le premier parcours complet.

## Développement local (2 process)

```bash
# 1) backend
cd backend && PORT=5001 python api.py

# 2) frontend (pointe sur le backend local)
cd frontend && API_BASE=http://localhost:5001 PORT=8000 python app.py
```

Ouvre http://127.0.0.1:8000 (la caméra exige un contexte sécurisé : `127.0.0.1`
est accepté ; pour un test mobile via IP réseau, servir en HTTPS).

Modes test : dans **Réglages → Mode test**, active « Corps de test » (greffe
`corps.glb`, sans Meshy) et/ou « Visage de test » (greffe `visage.glb`, ignore le
visage capturé). Désactivés par défaut ; nécessite que `corps.glb`/`visage.glb`
soient présents côté backend.

## Notes démo

- **Poids des avatars** : un avatar fait ~60 Mo (corps haute résolution + textures).
  IndexedDB tient pour quelques avatars ; peut être purgé par le navigateur sous
  pression de stockage (surtout iOS). D'où le bouton **Télécharger** (copie durable).
  Levier d'allègement : Meshy `should_remesh` + texture réduite.
- **Protection crédits Meshy** : volontairement absente (démo). À rajouter (BFF /
  gate) avant une mise en prod ouverte.
- **Cold start** : la 1re requête après inactivité charge rembg+mediapipe
  (~10-30 s). Mettre `min-scale=1` pendant une démo live si gênant.
