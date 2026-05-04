#!/bin/bash
#SBATCH --account=e52938
#SBATCH --partition=gengpu
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --job-name=babyai_bench
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail
set -x

REPO_ROOT="/gpfs/home/rvt4270/cs496/pilot_project/AgentGym-RL"
cd "${REPO_ROOT}"

module purge || true
source /hpc/software/mamba/23.1.0/etc/profile.d/conda.sh
conda activate agentgym-rl

export TOKENIZERS_PARALLELISM=false
export VERL_ATTN_IMPLEMENTATION="${VERL_ATTN_IMPLEMENTATION:-sdpa}"
export WANDB_MODE=offline
export ENV_SERVER_URL="http://127.0.0.1:36001"
export AGENT_MODEL_PATH="models/Qwen2.5-0.5B-Instruct"
export EXP_NAME="babyai_benchmark_${SLURM_JOB_ID}"
export MODEL_SAVE_PATH="saves/${EXP_NAME}"
export TRAIN_FILE="AgentGym-RL-Data-ID/train/babyai_train.json"

# Keep this conservative for the first benchmark run. Increase after it completes.
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
export ROLLOUT_N="${ROLLOUT_N:-2}"
export ROUNDS="${ROUNDS:-8}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}"

mkdir -p slurm_logs "${MODEL_SAVE_PATH}"

date
hostname
nvidia-smi
python --version
python -m pip show torch vllm transformers tokenizers agentenv agentenv_babyai | sed -n '1,220p'

babyai --host 127.0.0.1 --port 36001 > "${MODEL_SAVE_PATH}/babyai_server.log" 2>&1 &
BABYAI_PID=$!
trap 'kill ${BABYAI_PID} 2>/dev/null || true' EXIT

for i in $(seq 1 120); do
  if python - <<'PY'
import requests
requests.get("http://127.0.0.1:36001/", timeout=2).raise_for_status()
PY
  then
    break
  fi
  sleep 2
done

python - <<'PY'
import requests
print(requests.get("http://127.0.0.1:36001/", timeout=5).text)
PY

bash examples/train/AgentGym-RL/babyai_smoke_local.sh

date
