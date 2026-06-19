# 🛠️ Commandes utiles

Mémo personnel — j'ajoute des sections au fur et à mesure.

---

## 🐳 Docker

### Comprendre les 4 notions de base
| Notion | C'est quoi | Analogie |
|--------|-----------|----------|
| **Image** | Un modèle figé, prêt à lancer (ex. `mytwin-unitex:local`) | La recette + les ingrédients |
| **Conteneur** | Une image en train de tourner | Le plat cuisiné |
| **Volume** | Un stockage qui survit aux conteneurs (ex. les poids) | Le frigo |
| **Cache de build** | Couches intermédiaires réutilisées entre 2 builds | Les préparations d'avance |

### Docker Compose (orchestrer plusieurs conteneurs)
```bash
docker compose up -d --build        # build (si besoin) + démarre en arrière-plan (-d)
docker compose up -d --build unitex # ne (re)build/démarre QUE le service unitex
docker compose up -d                # démarre sans rebuild (réutilise les images)
docker compose down                 # arrête et supprime les conteneurs (garde les volumes)
docker compose down -v              # ⚠️ + supprime les volumes (efface les poids !)
docker compose restart unitex       # redémarre un service
docker compose ps                   # liste les services et leur état
docker compose logs -f              # suit les logs de tout (Ctrl-C pour sortir)
docker compose logs -f unitex       # suit les logs d'un seul service
docker compose logs --tail=50 unitex# les 50 dernières lignes seulement
```

### Conteneurs
```bash
docker ps                           # conteneurs en cours
docker ps -a                        # + ceux arrêtés
docker exec -it geneman bash        # entrer DANS le conteneur (shell interactif)
docker exec geneman nvidia-smi      # lancer une commande ponctuelle dedans
docker stop geneman                 # arrêter
docker rm geneman                   # supprimer (doit être arrêté)
docker logs -f geneman              # logs d'un conteneur précis
```

### Images
```bash
docker images                       # liste les images locales
docker rmi mytwin-unitex:local      # supprimer une image
docker image prune -f               # supprimer les images "dangling" (orphelines)
docker image prune -af              # supprimer TOUTES les images non utilisées
```

### Volumes (où vivent les poids)
```bash
docker volume ls                    # liste les volumes
docker volume inspect mytwin-avatar_geneman_weights   # détails (chemin sur disque)
```

### 🧹 Nettoyage / espace disque (quand « No space left on device »)
```bash
df -h /                             # espace libre sur le disque système
docker system df                    # ce que Docker occupe (images/conteneurs/cache)
docker builder prune -af            # vide TOUT le cache de build (récupère le plus)
docker system prune -af             # cache + images + conteneurs arrêtés inutilisés
# ⚠️ ne JAMAIS faire `prune --volumes` ici : ça effacerait les poids téléchargés
```

### GPU dans Docker
```bash
nvidia-smi                          # état du GPU sur l'hôte
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi  # test GPU en conteneur
docker exec geneman nvidia-smi      # voir le GPU vu depuis le conteneur
```

### Débogage rapide
```bash
docker inspect geneman              # config complète (réseau, montages, env…) en JSON
docker compose config               # affiche le docker-compose.yml "résolu" (avec .env)
docker stats                        # conso CPU/RAM/GPU-mem en temps réel par conteneur
```

---

## 🌐 Tester les API (depuis l'instance)
```bash
curl http://localhost:8001/health   # GeneMAN
curl http://localhost:8002/health   # UniTEX
```

---

## 🔧 Git (rappel)
```bash
git pull                            # récupérer les derniers changements
git status                          # voir l'état local
git log --oneline -5                # 5 derniers commits
```
