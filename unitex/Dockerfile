# UniTEX — env reproduit depuis le env.sh officiel du repo.
# Base CUDA 12.1 "devel" : nvcc de CUDA 11.8 plante (segfault) en générant du code
# Hopper (sm_90) pour les kernels nvdiffrast/slangtorch -> CUDA 12.x est requis pour H100.
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_CUDA_ARCH_LIST="9.0" \
    FORCE_CUDA=1 \
    HF_HOME=/weights/hf_cache

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-dev python3-pip \
        git wget curl ninja-build build-essential cmake \
        libgl1 libglib2.0-0 libegl1 libgles2 libglvnd-dev \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel

# --- PyTorch 2.4.1 (CUDA 12.1) ---
RUN pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
        --index-url https://download.pytorch.org/whl/cu121

# --- Kaolin (wheel officiel cu121 pour torch 2.4.1) ---
RUN pip install kaolin==0.17.0 \
        -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.1_cu121.html

# --- Dépendances (reproduit env.sh, variantes cu12) ---
RUN pip install xformers==0.0.28 --index-url https://download.pytorch.org/whl/cu121
RUN pip install transformers diffusers bitsandbytes==0.45 tokenizers \
    && pip install tqdm ftfy regex safetensors thop opencv-python opencv-contrib-python datetime bypy lpips matplotlib imageio tensorboard \
    && pip install datasets accelerate einops h5py \
    && pip install calflops OmegaConf icecream \
    && pip install pymcubes==0.1.6 pymeshlab==2023.12.post3 onnxruntime onnxruntime-gpu \
    && pip install trimesh==3.20.2 pyhocon deepspeed \
    && pip install rembg fpsample triton==3.0.0 open3d xatlas==0.0.10 \
    && pip install gpytoolbox==0.3.3 cupy-cuda12x==13.4.1 \
    && pip install async-timeout==5.0.1 timeout-decorator==0.5.0 peft==0.15.2 \
    && pip install jaxtyping==0.3.1 sentencepiece==0.2.0 timm==0.6.13 kornia==0.8.0 \
    && pip install pyexr==0.5.0 imageio==2.37 imageio-ffmpeg==0.4.7

# nvdiffrast + slangtorch (1.3.4 = version stable conseillée par le repo)
RUN pip install git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation
RUN pip install slangtorch==1.3.4
RUN pip install git+https://github.com/thomgrand/torch_kdtree.git --no-build-isolation

# --- Code UniTEX ---
WORKDIR /app
RUN git clone --depth 1 https://github.com/YixunLiang/UniTEX.git

# Serveur API
RUN pip install "huggingface_hub[cli]" fastapi "uvicorn[standard]" python-multipart

COPY api.py /app/UniTEX/api.py
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

VOLUME ["/weights"]
EXPOSE 8000
ENTRYPOINT ["/entrypoint.sh"]
