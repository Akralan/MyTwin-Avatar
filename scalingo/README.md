# MyTwin Avatar — Web app Scalingo (Meshy AI)

Web app **mobile-first** : une photo → un modèle 3D (via l'API Meshy AI), affiché
avec `<model-viewer>` (orbite, pinch-zoom, AR sur mobile). Pensée pour une **démo
client** : UI propre + protections.

## 🔐 Sécurité (déjà en place)
- **Clé API Meshy 100 % côté serveur** — jamais envoyée au navigateur.
- **Rate-limiting par IP** sur la génération (`RATE_LIMIT_GENERATE`) → protège tes crédits.
- **Code d'accès optionnel** (`ACCESS_CODE`) → réserve l'app à tes clients.
- **Images validées + ré-encodées + redimensionnées** (Pillow) ; taille d'upload limitée.
- **En-têtes de sécurité** (CSP, `X-Content-Type-Options`, `X-Frame-Options`…).
- **HTTPS** automatique côté Scalingo.

## 🚀 Déploiement

### 1. Créer l'app + variables d'env
```bash
scalingo create mytwin-avatar
scalingo --app mytwin-avatar env-set MESHY_API_KEY=msy_xxxxx
scalingo --app mytwin-avatar env-set SECRET_KEY=$(openssl rand -hex 32)
# optionnels :
scalingo --app mytwin-avatar env-set ACCESS_CODE=monCodeClient
scalingo --app mytwin-avatar env-set RATE_LIMIT_GENERATE="8 per hour"
```

### 2. Pousser le code
Ce dossier `scalingo/` doit être **la racine** déployée. Comme il est dans un
sous-dossier du dépôt, utilise un `git subtree` :

```bash
# depuis la racine du dépôt
scalingo --app mytwin-avatar git-remote          # ajoute le remote 'scalingo'
git subtree push --prefix scalingo scalingo master
```

> Alternative : copier le contenu de `scalingo/` dans un dépôt dédié et
> `git push scalingo master`.

### 3. C'est en ligne
`https://mytwin-avatar.osc-fr1.scalingo.io` (ton URL Scalingo).

## ⚙️ Variables d'environnement
| Variable | Déf. | Rôle |
|----------|------|------|
| `MESHY_API_KEY` | — | **obligatoire** — clé API Meshy |
| `SECRET_KEY` | aléatoire | secret sessions (à fixer pour ne pas invalider les sessions au redéploiement) |
| `ACCESS_CODE` | *(vide)* | code d'accès partagé ; vide = accès libre |
| `MESHY_POSE_MODE` | `a-pose` | pose quand l'option A-pose est active |
| `RATE_LIMIT_GENERATE` | `8 per hour` | quota de génération par IP |
| `MAX_UPLOAD_MB` | `12` | taille max d'upload |
| `MESHY_TARGET_POLYCOUNT` | `30000` | densité du mesh |

## 🧪 Tester en local
```bash
cd scalingo
pip install -r requirements.txt
export MESHY_API_KEY=msy_xxxxx        # PowerShell : $env:MESHY_API_KEY="msy_xxxxx"
python app.py                          # http://localhost:5000
```

## ⚠️ Limites à connaître
- **1 worker** (Procfile) : l'état des jobs est en mémoire. Pour **scaler à
  plusieurs dynos/workers**, il faudra un stockage partagé (Redis) pour les jobs
  + le rate-limit, et un **object storage** (S3) pour les `.glb` (le disque
  Scalingo est éphémère et non partagé).
- Chaque génération **consomme des crédits Meshy**.
```
