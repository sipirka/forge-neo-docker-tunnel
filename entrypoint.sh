#!/usr/bin/env bash
set -e

export COMMANDLINE_ARGS="\
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
--loglevel WARNING \
--lowvram \
--mmap-torch-files \
--no-hashing \
--nunchaku \
--onnxruntime-gpu \
--pin-shared-memory \
--reserve-vram 2 \
--sage \
--sage-function fp16_cuda \
--skip-prepare-environment \
--skip-python-version-check \
--skip-torch-cuda-test \
--skip-version-check \
--theme dark \
--xformers"
#--force-upcast-attention \
#--sage2-function fp16_triton \
#--sage2-function fp16_cuda \
#--sage2-function fp8_cuda \
#--sage2-function fp8_cuda++ \
#--tiled-conv2d 64 \
#--tiled-conv2d 128 \
#--tiled-conv2d 256 \
#--tiled-conv2d 512 \

# 公式 webui.sh が参照する VENV_DIR / PYTHON をここで確定させる
export SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export VENV_DIR="${SCRIPT_DIR}/venv"
export PYTHON="${VENV_DIR}/bin/python3.13"

PORT="${PORT:-7860}"

# UV（高速パッケージマネージャ）の並列化設定
export UV_CONCURRENT_DOWNLOADS="${UV_CONCURRENT_DOWNLOADS:-16}"
export UV_CONCURRENT_INSTALLS="${UV_CONCURRENT_INSTALLS:-16}"

# pipバージョン確認とキャッシュ無効化（容量節約）
export UV_NO_CACHE=1
export PIP_NO_CACHE_DIR=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PYTHONDONTWRITEBYTECODE=1

# PYTORCH_ALLOC_CONF を設定（メモリ割り当て最適化）
export PYTORCH_ALLOC_CONF="garbage_collection_threshold:0.7,max_split_size_mb:128,expandable_segments:True"
# export PYTORCH_CUDA_ALLOC_CONF="garbage_collection_threshold:0.7,max_split_size_mb:128,expandable_segments:True"

# --- LIBRARY_PATH ---
export LIBRARY_PATH="${LIBRARY_PATH:-/usr/local/cuda-13.0/targets/x86_64-linux/lib/stubs}"
if [[ "${LIBRARY_PATH}" == *: ]]; then
    LIBRARY_PATH="${LIBRARY_PATH%:}"
    export LIBRARY_PATH
fi
echo "LIBRARY_PATH:   ${LIBRARY_PATH}"

# --- LD_PRELOAD ---
export LD_PRELOAD="${LD_PRELOAD:-/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4}"
# export LD_PRELOAD="/usr/local/cuda-13.0/targets/x86_64-linux/lib/libcusparse.so.12:${LD_PRELOAD}"
if [[ "${LD_PRELOAD}" == *: ]]; then
    LD_PRELOAD="${LD_PRELOAD%:}"
    export LD_PRELOAD
fi
echo "LD_PRELOAD:   ${LD_PRELOAD}"

# 作業ディレクトリへ移動
cd "$SCRIPT_DIR"

# venv の健全性チェック（公式 webui.sh と同条件: ディレクトリの有無ではなく実行可能な python の有無で判定）
if [ ! -x "${VENV_DIR}/bin/python" ]; then
    echo "[Entrypoint] venv が存在しない/不完全なため作成します..."
    uv venv "$VENV_DIR" --python 3.13 --seed
fi

# webui.sh を呼ぶ前に自分で venv を有効化（SageAttention チェックのため）
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

if ! python -c "import sageattention" 2>/dev/null; then
    echo "[Entrypoint] Warning: sageattention not found in image, this should not happen at runtime"
fi

# venv は用意・有効化済みなので、webui.sh 側の venv 処理（pip自動アップグレード等）をスキップさせる
export SKIP_VENV=1

# webui.sh を実行（exec で PID1 を置き換え、docker stop の SIGTERM が python まで確実に届くようにする）
exec bash ./webui.sh --listen --port "$PORT" "$@"
