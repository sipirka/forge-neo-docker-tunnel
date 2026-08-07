# syntax=docker/dockerfile:1
# build stage (CUDA 13.0)

ARG CUDA_VERSION=13.0.3
ARG SAGEATTENTION_VERSION=2.2.0
ARG SPARGEATTN_REF=ae5b629ebb41e41f86b3ea2ab5a3283f13ac151a
ARG FLASH_ATTN_REF=v2.8.3.post1
ARG NUNCHAKU_REF=v1.2.1
ARG TORCH_CUDA_ARCH_LIST="8.9"
ARG MAX_JOBS=16
ARG UV_CONCURRENT_DOWNLOADS=16
ARG UV_CONCURRENT_INSTALLS=16
ARG UV_VERSION=0.12.1
ARG TORCH_VERSION=2.13.0
ARG TORCH_INDEX=cu130
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/${TORCH_INDEX}
ARG FLASH_ATTN_VERSION=2.8.3+cu130torch2.13

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu24.04 AS builder

ARG SAGEATTENTION_VERSION
ARG SPARGEATTN_REF
ARG FLASH_ATTN_REF
ARG NUNCHAKU_REF
ARG TORCH_CUDA_ARCH_LIST
ARG MAX_JOBS
ARG UV_CONCURRENT_DOWNLOADS
ARG UV_CONCURRENT_INSTALLS
ARG UV_VERSION
ARG TORCH_VERSION
ARG TORCH_INDEX
ARG TORCH_INDEX_URL
ARG FLASH_ATTN_VERSION

ENV DEBIAN_FRONTEND=noninteractive \
    FORCE_CUDA=1 \
    TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
    MAX_JOBS="${MAX_JOBS}" \
    UV_CONCURRENT_DOWNLOADS="${UV_CONCURRENT_DOWNLOADS}" \
    UV_CONCURRENT_INSTALLS="${UV_CONCURRENT_INSTALLS}" \
    UV_PYTHON_DOWNLOADS=automatic \
    UV_NO_CACHE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV="/tmp/build/venv" \
    PATH="/tmp/build/venv/bin:/root/.local/bin:$PATH" \
    PYTHON="/tmp/build/venv/bin/python3.13"

COPY --from=uv /uv /usr/local/bin/uv

RUN apt update && \
    apt install -y \
    build-essential python3-dev libcairo2-dev pkg-config git curl wget && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

WORKDIR /tmp/build

RUN uv venv venv --python 3.13 && \
    . ./venv/bin/activate && \
    uv pip install wheel pip setuptools packaging ninja triton && \
    uv pip install "torch==${TORCH_VERSION}" torchvision --index-url "${TORCH_INDEX_URL}"

# SageAttention
RUN mkdir -p /tmp/wheels && \
    . ./venv/bin/activate && \
    git clone --branch "v${SAGEATTENTION_VERSION}" --depth=1 https://github.com/thu-ml/SageAttention.git && \
    cd SageAttention && \
    python setup.py bdist_wheel && \
    cp dist/*.whl /tmp/wheels/

# SpargeAttn
RUN . ./venv/bin/activate && \
    mkdir SpargeAttn && cd SpargeAttn && \
    git init -q && \
    git remote add origin https://github.com/thu-ml/SpargeAttn.git && \
    git fetch --depth=1 origin "${SPARGEATTN_REF}" && \
    git checkout -q FETCH_HEAD && \
    sed -i '/"-Xcompiler", "-include,cassert"/d' setup.py && \
    python setup.py bdist_wheel && \
    cp dist/*.whl /tmp/wheels/

# FlashAttention
RUN . ./venv/bin/activate && \
    curl -fL -o "/tmp/wheels/flash_attn-${FLASH_ATTN_VERSION}-cp313-cp313-linux_x86_64.whl" \
        "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.47/flash_attn-${FLASH_ATTN_VERSION}-cp313-cp313-linux_x86_64.whl" && \
    pip wheel --no-deps einops -w /tmp/wheels

# Nunchaku
RUN . ./venv/bin/activate && \
    pip install build && \
    git clone --recurse-submodules --shallow-submodules --depth=1 \
        --branch "${NUNCHAKU_REF}" https://github.com/nunchaku-tech/nunchaku.git && \
    cd nunchaku && \
    MAX_JOBS=4 NUNCHAKU_INSTALL_MODE=ALL NUNCHAKU_BUILD_WHEELS=1 python -m build --wheel --no-isolation && \
    cp dist/*.whl /tmp/wheels/

# onnxruntime-gpu wheel
RUN . ./venv/bin/activate && \
    pip wheel --no-deps --pre --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ort-cuda-13-nightly/pypi/simple/ onnxruntime-gpu -w /tmp/wheels



# multi stage build

FROM nvidia/cuda:${CUDA_VERSION}-cudnn-runtime-ubuntu24.04 AS minimal

ARG SAGEATTENTION_VERSION
ARG TORCH_CUDA_ARCH_LIST
ARG MAX_JOBS
ARG UV_CONCURRENT_DOWNLOADS
ARG UV_CONCURRENT_INSTALLS
ARG UV_VERSION
ARG TORCH_VERSION
ARG TORCH_INDEX
ARG TORCH_INDEX_URL

ENV DEBIAN_FRONTEND=noninteractive \
    TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" \
    MAX_JOBS="${MAX_JOBS}" \
    UV_CONCURRENT_DOWNLOADS="${UV_CONCURRENT_DOWNLOADS}" \
    UV_CONCURRENT_INSTALLS="${UV_CONCURRENT_INSTALLS}" \
    UV_PYTHON_DOWNLOADS=automatic \
    UV_NO_CACHE=1 \
    UV_CONSTRAINT="/etc/pip/constraints.txt" \
    PIP_CONSTRAINT="/etc/pip/constraints.txt" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV="/app/sd-webui-forge-neo/venv" \
    PATH="/app/sd-webui-forge-neo/venv/bin:/home/ubuntu/.local/bin:$PATH" \
    PYTHON="/app/sd-webui-forge-neo/venv/bin/python3.13" \
    LIBRARY_PATH="/usr/local/cuda-13.0/targets/x86_64-linux/lib/stubs:${LIBRARY_PATH}" \
    PYTORCH_ALLOC_CONF="garbage_collection_threshold:0.7,max_split_size_mb:128,expandable_segments:True"
    #PYTORCH_CUDA_ALLOC_CONF="garbage_collection_threshold:0.7,max_split_size_mb:128,expandable_segments:True"

RUN apt update && \
    apt install -y --no-install-recommends \
    sudo \
    cuda-driver-dev-13-0 build-essential ninja-build libcairo2 \
    ffmpeg \
    git curl wget libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 libgoogle-perftools4 libtcmalloc-minimal4 libcusparse12 xdg-utils bc aria2 && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

ENV LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4:${LD_PRELOAD}"
    #LD_PRELOAD="/usr/local/cuda-13.0/targets/x86_64-linux/lib/libcusparse.so.12:${LD_PRELOAD}"

RUN usermod -aG sudo ubuntu && \
    echo "ubuntu ALL=(ALL:ALL) NOPASSWD: ALL" >> /etc/sudoers.d/ubuntu && \
    chmod 0440 /etc/sudoers.d/ubuntu && \
    mkdir -p /etc/pip && \
    echo "torch==${TORCH_VERSION}" > /etc/pip/constraints.txt && \
    chmod 644 /etc/pip/constraints.txt

ADD https://api.github.com/repos/Haoming02/sd-webui-forge-classic/commits/neo /tmp/neo-rev.json

RUN mkdir -p /app && \
    cd /app && \
    git clone --depth=1 https://github.com/Haoming02/sd-webui-forge-classic sd-webui-forge-neo --branch neo && \
    cd sd-webui-forge-neo && \
    rm -rf .git && \
    mkdir -p extensions models output && \
    find . -maxdepth 1 -iname "requirements*.txt" -exec sed -i '/^setuptools==/d' {} \; && \
    rm -f /tmp/neo-rev.json

COPY --chown=ubuntu:ubuntu --chmod=755 ./entrypoint.sh /app/sd-webui-forge-neo/entrypoint.sh

RUN mkdir -p /home/ubuntu/.local/bin /home/ubuntu/.local/share/uv && \
    chown -R ubuntu:ubuntu /app /home/ubuntu/.local && \
    chmod -R 775 /app

USER ubuntu
WORKDIR /app/sd-webui-forge-neo

COPY --from=uv /uv /usr/local/bin/uv

RUN --mount=type=bind,from=builder,source=/tmp/wheels,target=/tmp/wheels,readonly \
    uv venv venv --python 3.13 --seed && \
    . ./venv/bin/activate && \
    uv pip install "torch==${TORCH_VERSION}" torchvision --index-url "${TORCH_INDEX_URL}" && \
    uv pip install svglib olefile ultralytics accelerate diffusers transformers peft protobuf sentencepiece huggingface-hub psutil && \
    uv pip install --no-index --find-links /tmp/wheels "sageattention==${SAGEATTENTION_VERSION}" spas-sage-attn flash-attn nunchaku && \
    uv pip install --prerelease=allow --find-links /tmp/wheels onnxruntime-gpu && \
    timeout 30m bash -c ". ./venv/bin/activate && ./webui.sh \
    --uv \
    --adv-samplers \
    --bf16-unet \
    --bf16-vae \
    --cuda-malloc \
    --cuda-stream \
    --enable-insecure-extension-access \
    --enable-triton-backend \
    --expandable-segments \
    --fast-fp8 \
    --fast-fp16 \
    --flash \
    --force-non-blocking \
    --force-xformers-vae \
    --fp16-text-enc  \
    --log-startup \
    --loglevel DEBUG \
    --lowvram \
    --mmap-torch-files \
    --no-hashing \
    --nunchaku \
    --onnxruntime-gpu \
    --pin-shared-memory \
    --reserve-vram 2 \
    --sage \
    --sage-function fp16_cuda \
    --skip-python-version-check \
    --skip-torch-cuda-test \
    --skip-version-check \
    --theme dark \
    --xformers \
    --exit" && \
    uv cache clean; \
    pip cache purge 2>/dev/null; \
    find /app/sd-webui-forge-neo/venv -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; \
    find /app/sd-webui-forge-neo/venv -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete 2>/dev/null; \
    find /app/sd-webui-forge-neo/venv -type d \( -name "tests" -o -name "test" -o -name "docs" -o -name "doc" -o -name "idle_test" \) -exec rm -rf {} + 2>/dev/null; \
    find /app/sd-webui-forge-neo/venv -type f -name "*.a" -delete 2>/dev/null; \
    find /app/sd-webui-forge-neo/venv -type f -name "*.so" -exec strip --strip-unneeded {} + 2>/dev/null; \
    rm -rf /home/ubuntu/.cache/* /var/lib/apt/lists/* /var/cache/apt/archives/* && \
    find /tmp -mindepth 1 -maxdepth 1 ! -name wheels -exec rm -rf {} +
    
EXPOSE 7860

ENTRYPOINT ["/app/sd-webui-forge-neo/entrypoint.sh"]
