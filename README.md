# MyTwin Avatar POC — GeneMAN + UniTEX (Docker GPU)

Deux modèles 3D dans des conteneurs séparés (stacks incompatibles), orchestrés par
Docker Compose. Objectif : sur une instance GPU Scaleway fraîche, **cloner + lancer**,
rien d'autre à faire.

| Modèle | Rôle | Stack |
|--------|------|-------|
| [GeneMAN](https://github.com/3DTopia/GeneMAN) | Reconstruction 3D humain depuis 1 image | Py3.10 · torch 2.1.2 · CUDA 12.1 |
| [UniTEX](https://github.com/YixunLiang/UniTEX) | Texturing haute fidélité de mesh 3D | Py3.10 · torch 2.4.1 · CUDA 12.1 |

> 💡 CUDA **12.1** (et non 11.8) car le H100 (Hopper/sm_90) fait planter nvcc 11.8
> lors de la compilation des kernels. Le driver de l'instance (CUDA 13) est compatible.

> ⚠️ Chaque modèle exige un GPU costaud (GeneMAN : **≥ 20 Go VRAM**). Prends une
> instance type **L40S / H100** chez Scaleway.

## Architecture

```
instance Scaleway ── git clone ── docker compose up -d --build
                                       │
                                       ├─ build des 2 images sur place (1 seule fois)
                                       └─ poids téléchargés depuis HuggingFace au 1er run
                                          (cachés dans un volume -> 1 seule fois)
```

> Option (non utilisée par défaut) : pré-builder vers GHCR via GitHub Actions pour
> éviter de recompiler sur une *nouvelle* instance — voir la fin du README.

## Déploiement sur l'instance Scaleway (chemin par défaut)

Les images sont **buildées directement sur l'instance**, une seule fois.

```bash
git clone https://github.com/Akralan/MyTwin-Avatar.git
cd MyTwin-Avatar
cp .env.example .env && nano .env        # renseigner HF_TOKEN
bash scripts/bootstrap_instance.sh       # Docker + toolkit GPU, puis build + up
```

> Sur les pages HuggingFace des modèles *gated*, **accepte les licences** avec ton
> compte avant le 1er démarrage (sinon le téléchargement des poids échoue).

> ⏱️ Le 1ʳᵉ `up --build` est **long** (compilations CUDA : kaolin, nvdiffrast,
> slangtorch). Les démarrages suivants réutilisent l'image (`docker compose up -d`).

Chaque modèle expose alors une **API HTTP** :

| Modèle | Endpoint | Port hôte |
|--------|----------|-----------|
| GeneMAN | `POST /reconstruct` (image) | `http://<IP>:8001` |
| UniTEX  | `POST /texture` (image + mesh) | `http://<IP>:8002` |

> 🔓 **Security group Scaleway** : ouvre les ports **8001** et **8002** en entrée
> (idéalement restreints à l'IP de ta machine).

### Flux pour ton app web (locale → instance)

```
image ─POST /reconstruct─▶ GeneMAN ─▶ job_id ─poll /jobs/{id}─▶ mesh.glb   (SANS texture)
mesh + image ─POST /texture─▶ UniTEX ─▶ job_id ─poll /jobs/{id}─▶ mesh.glb (AVEC texture)
```

Les deux API sont **asynchrones** (GeneMAN dure plusieurs minutes) : `POST` renvoie
un `job_id`, tu interroges `GET /jobs/{id}` jusqu'à `status: done`, puis tu récupères
le mesh via `result_url`. Affichage 3D côté navigateur avec `<model-viewer>` ou three.js.

Un client complet (upload → reconstruction → texturing → double affichage) est fourni
dans **`examples/client.html`** — ouvre-le dans un navigateur pour tester de bout en bout.

> ✅ Les commandes d'inférence sont **les vraies** (extraites des repos) : GeneMAN via
> `run_geneman.sh`, UniTEX via son `CustomRGBTextureFullPipeline`. Seuls les chemins
> d'export GeneMAN restent à confirmer au 1er run (cf. plus bas).

## Pipeline réel (vérifié dans les repos)

- **GeneMAN** = optimisation **multi-étapes** (`preprocess → geometry-init →
  geometry-sculpt → texture`), pilotée par `launch.py`. Le wrapper
  `geneman/run_geneman.sh` n'exécute que les **étapes géométrie** (1-2) et exporte
  `model.obj` → c'est ton mesh **sans texture**. La texture est déléguée à UniTEX.
- **UniTEX** = pipeline en mémoire (`CustomRGBTextureFullPipeline`), chargé **une
  fois** puis réutilisé. Prend `(image, mesh)` → mesh **avec texture**.

## Points d'attention (honnêteté technique)

- ⏱️ **GeneMAN est LENT** : la reconstruction par optimisation prend de
  **plusieurs dizaines de minutes à quelques heures par image** sur un seul GPU.
  D'où l'API asynchrone. À budgétiser pour le POC (coût GPU Scaleway à l'heure).
- **Chemins d'export GeneMAN** : `run_geneman.sh` reprend fidèlement `script/run.sh`
  du repo, mais les chemins exacts (`it5000-export/model.obj`, ckpts) sont à
  **confirmer au 1er run réel**. Tout est surchargeable via `GENEMAN_CMD` sans rebuild.
- **UniTEX** : env reproduit depuis le `env.sh` officiel (pas de pytorch3d — vérifié).
  slangtorch épinglé en **1.3.4** (version stable conseillée par le repo).
- **Compilation CUDA** (nvdiffrast, kaolin, slangtorch, torch_kdtree) : le 1ʳᵉ
  `up --build` est long. Builds suivants instantanés (image en cache local).
- **Modèles gated** : sans acceptation de licence + `HF_TOKEN`, le download échoue.

## Structure

```
├── docker-compose.yml          # 2 services GPU + build local + ports API + volumes
├── .env.example                # HF_TOKEN (REGISTRY/TAG = option registry)
├── geneman/{Dockerfile,entrypoint.sh,api.py,run_geneman.sh}  # API :8001
├── unitex/{Dockerfile,entrypoint.sh,api.py}                  # API :8002
├── examples/client.html        # client web démo (affichage avec/sans texture)
├── scripts/bootstrap_instance.sh
└── .github/workflows/build.yml # OPTIONNEL : pré-build vers GHCR (manuel)
```

## Option : pré-builder vers un registry (GHCR)

Utile seulement si tu veux **éviter de recompiler sur une nouvelle instance**.
Lance le workflow `Build & push images` manuellement (onglet **Actions**) : il
pousse `ghcr.io/akralan/mytwin-{geneman,unitex}`. Sur l'instance, remplace alors
les blocs `build:` du `docker-compose.yml` par `image: ghcr.io/...` et fais
`docker compose pull && docker compose up -d`.
