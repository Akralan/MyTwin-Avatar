# GeneMAN — base CUDA 11.8 "devel" (nvcc requis pour compiler les extensions CUDA)
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" \
    # Le cache HuggingFace pointe vers le volume monté -> pas de re-download au redémarrage
    HF_HOME=/weights/hf_cache

# --- Dépendances système ---
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3-pip \
        git wget curl ninja-build build-essential \
        libgl1 libglib2.0-0 libegl1 libgles2 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel

# --- PyTorch (CUDA 11.8) ---
RUN pip install torch==2.0.0 torchvision==0.15.1 \
        --index-url https://download.pytorch.org/whl/cu118

# --- Code GeneMAN ---
WORKDIR /app
RUN git clone --depth 1 https://github.com/3DTopia/GeneMAN.git
WORKDIR /app/GeneMAN
# NOTE : si requirements.txt entre en conflit avec la version de torch ci-dessus,
# c'est ici qu'il faudra ajuster (pin de versions).
RUN pip install -r requirements.txt || \
    (echo ">>> requirements.txt a échoué, à ajuster manuellement" && exit 1)

# Outil de téléchargement des poids + serveur API
RUN pip install "huggingface_hub[cli]" fastapi "uvicorn[standard]" python-multipart

COPY api.py /app/api.py
COPY run_geneman.sh /app/run_geneman.sh
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh /app/run_geneman.sh

EXPOSE 8000

# Volume où atterrissent tous les poids (persisté côté hôte via docker-compose)
VOLUME ["/weights"]

ENTRYPOINT ["/entrypoint.sh"]
