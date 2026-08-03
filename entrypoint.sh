#!/usr/bin/env bash
set -e

export COMMANDLINE_ARGS="\
--uv \
--adv-samplers \
--autotune \
--bf16-unet \
--bf16-vae \
--cuda-malloc \
--cuda-stream \
--enable-insecure-extension-access \
--fast-fp8 \
--fast-fp16 \
--force-non-blocking \
--force-xformers-vae \
--fp16-text-enc  \
--log-startup \
--loglevel DEBUG \
--mmap-torch-files \
--no-hashing \
--pin-shared-memory \
--reserve-vram 0.5 \
--sage \
--sage-function fp16_cuda \
--skip-prepare-environment \
--skip-python-version-check \
--skip-torch-cuda-test \
--skip-version-check \
--theme dark \
--xformers"
#--bnb \
#--flash \
#--force-non-blocking \
#--force-upcast-attention \
#--gpu-only \
#--novram  \
#--nunchaku \
#--onnxruntime-gpu \
#--sage2-function fp16_triton \

export SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export install_dir="$(cd "$SCRIPT_DIR/.." && pwd)"
export clone_dir="${SCRIPT_DIR##*/}"
export python_cmd="${install_dir}/${clone_dir}/venv/bin/python3"

PORT="${PORT:-7860}"

# MAX_JOBS を動的に設定（環境変数がなければ CPU コア数を使用）
export MAX_JOBS="${MAX_JOBS:-$(nproc)}"
echo "[Entrypoint] MAX_JOBS set to: $MAX_JOBS"

# UV（高速パッケージマネージャ）の並列化設定
export UV_CONCURRENT_DOWNLOADS="${UV_CONCURRENT_DOWNLOADS}"
export UV_CONCURRENT_INSTALLS="${UV_CONCURRENT_INSTALLS}"

# pipバージョン確認とキャッシュ無効化（容量節約）
export PIP_NO_CACHE_DIR=1
export PIP_DISABLE_PIP_VERSION_CHECK=1

# PYTORCH_ALLOC_CONF を設定（メモリ割り当て最適化）
# export LD_PRELOAD="/usr/local/cuda-13.0/targets/x86_64-linux/lib/libcusparse.so.12"
export PYTORCH_ALLOC_CONF="garbage_collection_threshold:0.7,max_split_size_mb:128"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export LIBRARY_PATH="/usr/local/cuda-13.0/targets/x86_64-linux/lib/stubs:${LIBRARY_PATH}"

# 作業ディレクトリへ移動
pushd /app/sd-webui-forge-neo

# venv の有無を確認（Docker の再起動時用）
if [ ! -d "venv" ]; then
    echo "[Entrypoint] venv が存在しないため作成します..."
    uv venv venv --python 3.13 --seed
fi

# webui.sh を実行 (exec で PID1 を置き換える)
source ./venv/bin/activate
if ! python -c "import sageattention" 2>/dev/null; then
    echo "[Entrypoint] SageAttentionをインストール中..."
    uv pip install sageattention
fi

bash ./webui.sh --listen --port "$PORT" "$@"

popd
