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
scw container container update <container-id> \
    registry-image=rg.fr-par.scw.cloud/<namespace>/mytwin-api:latest
```

Réglages recommandés du Serverless Container :

| Réglage | Valeur | Pourquoi |
|---|---|---|
| Port | `8080` | port exposé par le `Dockerfile` |
| RAM / vCPU | **3072 MB** / **~2500 mVCPU** | le blending de texture HD + rembg + mediapipe peut dépasser 2 Go → OOM sinon ; 1024 MB = échec quasi certain |
| Concurrency | **1** | une greffe par instance (2 greffes simultanées se font OOM) ; laisser Scaleway scaler en plusieurs instances |
| Timeout requête | **300 s** | la greffe est synchrone (~1 min CPU) ; couvrir cold start + greffe |
| Privacy | **Public** | le navigateur appelle l'API directement ; un conteneur privé bloque tout |
| Variables | `MESHY_API_KEY`, `CORS_ORIGIN=https://<frontend>` | clé Meshy serveur ; origine autorisée (voir Dépannage) |

Scale-to-zero : la 1re requête après inactivité paie le cold start (chargement
rembg + mediapipe). Le tag `:latest` étant mutable, un `docker push` suivi d'un
redéploiement du conteneur suffit pour livrer une nouvelle version.

En pratique, ce build/push est **automatisé par GitHub Actions**
(`.github/workflows/deploy-backend.yml`) : à chaque push sur `main` touchant
`scalingo/backend/**`, les runners GitHub construisent l'image et la poussent sur
le registre (pas d'upload depuis une machine locale). Seul secret requis :
`SCW_SECRET_KEY` (clé API Scaleway dédiée, droits limités au Container Registry).
Les `.glb` de test sont versionnés (exception dans `.gitignore`) pour que la CI
puisse les embarquer. Le redéploiement du conteneur reste manuel (ou via le bloc
optionnel commenté dans le workflow, une fois le conteneur créé).

## Frontend (`frontend/`) — Scalingo

Flask minimal (Flask + gunicorn) : sert `index.html`, `models/face_landmarker.task`
(asset client) et injecte l'URL du backend via `API_BASE`. Aucun secret, aucun
compute. La galerie et le parcours sont 100 % côté navigateur.

Variables :

| Variable | Rôle | Prod |
|---|---|---|
| `API_BASE` | URL du backend Scaleway (`https://…scw.cloud`) que le navigateur appelle | **requise** ; doit inclure `https://` sinon la CSP bloque |
| `PROXY_API` | **dev uniquement** : relaie les appels en same-origin (évite CORS/HTTPS local) | **laisser vide** |
| `PORT` | port d'écoute | **ne pas définir** : Scalingo l'injecte |

Aucun secret côté frontend (la clé Meshy est 100 % côté backend).

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

## Dépannage

- **Frontend « Hors ligne » / « Failed to fetch » alors que `curl <backend>/healthz`
  répond** : le navigateur est bloqué là où curl ne l'est pas → **CORS** ou **CSP**.
  Ouvrir la console (F12) :
  - `... blocked by CORS policy` → `CORS_ORIGIN` du backend ne matche pas l'origine
    du frontend **au caractère près**. Pièges : **slash final**
    (`https://x.scalingo.io/` ≠ `https://x.scalingo.io`), `http` vs `https`, ou
    l'URL du backend mise par erreur. Débloquer avec `CORS_ORIGIN=*` puis
    redéployer ; durcir ensuite avec l'URL exacte du frontend.
  - `Refused to connect ... Content Security Policy ... connect-src` → `API_BASE`
    du frontend mal formée (souvent le `https://` manquant). La corriger et
    redéployer le frontend.
- **Le conteneur ne se met pas à jour après un nouveau `:latest`** : un push d'image
  ne redéploie PAS un conteneur en place. Cliquer *Deploy* dans la console, ou
  activer le bloc auto-redeploy commenté dans `.github/workflows/deploy-backend.yml`
  (nécessite `SCW_ACCESS_KEY`, `SCW_DEFAULT_PROJECT_ID` et le `CONTAINER_ID`).
- **Push CI en échec** : au *login* → secret `SCW_SECRET_KEY` absent/erroné ; au
  *push* → le namespace de registre (`fr-par`) n'existe pas encore.
- **Caméra inactive** : elle exige un contexte sécurisé (HTTPS, ou `127.0.0.1` en
  local). Vérifier aussi l'en-tête `Permissions-Policy: camera=(self)`.

## Notes démo

- **Poids des avatars** : un avatar fait ~60 Mo (corps haute résolution + textures).
  IndexedDB tient pour quelques avatars ; peut être purgé par le navigateur sous
  pression de stockage (surtout iOS). D'où le bouton **Télécharger** (copie durable).
  Levier d'allègement : Meshy `should_remesh` + texture réduite.
- **Protection crédits Meshy** : volontairement absente (démo). À rajouter (BFF /
  gate) avant une mise en prod ouverte.
- **Cold start** : la 1re requête après inactivité charge rembg+mediapipe
  (~10-30 s). Mettre `min-scale=1` pendant une démo live si gênant.
