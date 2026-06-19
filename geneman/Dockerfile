# GeneMAN — base CUDA 12.1 "devel" : nvcc de CUDA 11.8 plante (segfault) en générant
# du code Hopper (sm_90) pour nerfacc/tiny-cuda-nn/nvdiffrast -> CUDA 12.x requis pour H100.
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # H100 = sm_90 uniquement -> compilation CUDA bien plus rapide
    TORCH_CUDA_ARCH_LIST="9.0" \
    TCNN_CUDA_ARCHITECTURES="90" \
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

# --- PyTorch (CUDA 12.1) — 2.1.2 = version la plus proche du testé (2.0.0)
#     disposant de wheels cu121, requise pour Hopper/sm_90. ---
RUN pip install torch==2.1.2 torchvision==0.16.2 \
        --index-url https://download.pytorch.org/whl/cu121

# --- Code GeneMAN ---
WORKDIR /app
RUN git clone --depth 1 https://github.com/3DTopia/GeneMAN.git
WORKDIR /app/GeneMAN

# Plusieurs paquets (nerfacc, tiny-cuda-nn, nvdiffrast, envlight, pysdf, PyMCubes)
# importent torch/numpy DANS leur setup.py -> incompatible avec l'isolation de build
# de pip (env de build sans torch => "No module named 'torch'").
# Solution : pré-installer les deps de build puis installer SANS isolation
# (torch est déjà présent dans l'image).
RUN pip install ninja Cython pybind11 "numpy==1.26.4"
RUN pip install --no-build-isolation -r requirements.txt

# Serveur API : on réutilise fastapi (0.88) et huggingface_hub (0.21.4) déjà
# épinglés par GeneMAN ; on ajoute juste uvicorn + multipart + extras CLI.
RUN pip install "huggingface_hub[cli]==0.21.4" "uvicorn[standard]" python-multipart

COPY api.py /app/api.py
COPY run_geneman.sh /app/run_geneman.sh
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh /app/run_geneman.sh

EXPOSE 8000

# Volume où atterrissent tous les poids (persisté côté hôte via docker-compose)
VOLUME ["/weights"]

ENTRYPOINT ["/entrypoint.sh"]
