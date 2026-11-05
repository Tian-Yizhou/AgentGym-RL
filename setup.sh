#!/bin/bash
# Quick-start environment setup for AgentGym-RL + TextWorldExpress.
#
# Run from the repository root. Targets (default: rl twx data):
#   rl     conda env `agentgym-rl`: RL trainer (verl fork, vLLM) + AgentGym clients.     Needs an NVIDIA GPU to *run*.
#   twx    conda env `agentenv-textworldexpress`: TextWorldExpress env server (Java from conda-forge).
#   data   item-id files for TextWorldExpress (AgentGym-RL/AgentItemId, AgentGym-RL/AgentEval).
#   model  download $MODEL (default Qwen/Qwen2.5-7B-Instruct, ~15 GB) to AgentGym-RL/models/.
#
#   bash setup.sh                 # rl + twx + data
#   bash setup.sh model           # just the model
#   bash setup.sh rl twx data model
#
# Overridable: RL_ENV (agentgym-rl), TWX_ENV (agentenv-textworldexpress),
#              TORCH_CUDA (cu121; cu124 needs NVIDIA driver >= 550), MODEL (Qwen/Qwen2.5-7B-Instruct).
#
# Version notes (why the pins):
#   * verl's vLLM integration supports vllm <= 0.6.3, which requires torch == 2.4.0 exactly.
#   * flash-attn 2.7.3 wheel built for torch 2.4 (the torch 2.8 wheels need GLIBC >= 2.32).
#   * transformers 4.51.3 (AgentGym-RL); datasets < 4 and trl 0.19.1 are the newest releases
#     compatible with it (newer ones need huggingface-hub >= 1.0 / transformers >= 4.56).
set -eo pipefail

RL_ENV="${RL_ENV:-agentgym-rl}"
TWX_ENV="${TWX_ENV:-agentenv-textworldexpress}"
TORCH_CUDA="${TORCH_CUDA:-cu121}"
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
FLASH_ATTN_WHEEL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"
TARGETS=("$@")
[ ${#TARGETS[@]} -eq 0 ] && TARGETS=(rl twx data)

log() { echo -e "\n\033[1;34m[setup] $*\033[0m"; }

source "$(conda info --base)/etc/profile.d/conda.sh"

env_exists() { conda env list | awk '{print $1}' | grep -qx "$1"; }

ensure_submodule() {
  if [ ! -f AgentGym/agentenv/pyproject.toml ]; then
    log "Fetching the AgentGym submodule"
    git submodule update --init --recursive
  fi
}

setup_rl() {
  ensure_submodule
  if env_exists "${RL_ENV}"; then
    log "Reusing existing conda env ${RL_ENV}"
  else
    log "Creating conda env ${RL_ENV} (python 3.10)"
    conda create -n "${RL_ENV}" python=3.10 -y
  fi
  conda activate "${RL_ENV}"

  log "Installing torch 2.4.0 (${TORCH_CUDA})"
  pip install torch==2.4.0 torchvision==0.19.0 --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"

  log "Installing flash-attn 2.7.3 (torch 2.4 wheel)"
  pip install "${FLASH_ATTN_WHEEL}"

  log "Installing AgentGym-RL (verl) and the AgentGym client library (agentenv)"
  pip install -e "${ROOT}/AgentGym-RL"
  pip install -e "${ROOT}/AgentGym/agentenv"
  pip install "transformers==4.51.3" "datasets>=3,<4" "trl==0.19.1"

  log "Checking the ${RL_ENV} env"
  python - <<'EOF'
import torch, vllm, flash_attn, flash_attn.bert_padding, transformers, datasets
import verl
from agentenv.envs import TextWorldExpressEnvClient
from verl.utils.agentgym.client import init_env_client
print(f"torch {torch.__version__} | vllm {vllm.__version__} | flash_attn {flash_attn.__version__} | "
      f"transformers {transformers.__version__} | datasets {datasets.__version__} | GPU visible: {torch.cuda.is_available()}")
EOF
  pip check || echo "[setup] warning: pip reports dependency conflicts (see above)"
  conda deactivate
}

setup_twx() {
  ensure_submodule
  log "Setting up the TextWorldExpress env server (${TWX_ENV})"
  ENV_NAME="${TWX_ENV}" bash "${ROOT}/AgentGym/agentenv-textworldexpress/setup.sh"
}

setup_data() {
  ensure_submodule
  log "Generating TextWorldExpress item ids"
  python3 "${ROOT}/AgentGym/agentenv-textworldexpress/scripts/make_data.py" --out-dir "${ROOT}/AgentGym-RL"
}

setup_model() {
  local dir="${ROOT}/AgentGym-RL/models/$(basename "${MODEL}")"
  log "Downloading ${MODEL} -> ${dir}"
  if env_exists "${RL_ENV}"; then conda activate "${RL_ENV}"; fi
  python -c "from huggingface_hub import snapshot_download; snapshot_download('${MODEL}', local_dir='${dir}')"
}

for target in "${TARGETS[@]}"; do
  case "${target}" in
    rl) setup_rl ;;
    twx) setup_twx ;;
    data) setup_data ;;
    model) setup_model ;;
    *) echo "Unknown target '${target}'. Use: rl twx data model" >&2; exit 2 ;;
  esac
done

log "Done. Next steps:
  conda activate ${TWX_ENV} && textworldexpress --host 127.0.0.1 --port 36005     # env server (keep running)
  conda activate ${RL_ENV}  && bash examples/train/AgentGym-RL/textworldexpress_train.sh"
