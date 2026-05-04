#!/bin/bash
#SBATCH --account=e52938
#SBATCH --partition=gengpu
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --job-name=twx_bench
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail
set -x

REPO_ROOT="/gpfs/home/rvt4270/cs496/pilot_project/AgentGym-RL"
cd "${REPO_ROOT}"

module purge || true
module load "${JAVA_MODULE:-java/jdk-17.0.2+8}"

source /hpc/software/mamba/23.1.0/etc/profile.d/conda.sh
conda activate agentgym-rl

export TOKENIZERS_PARALLELISM=false
export VERL_ATTN_IMPLEMENTATION="${VERL_ATTN_IMPLEMENTATION:-sdpa}"
export WANDB_MODE=offline
export ENV_SERVER_URL="http://127.0.0.1:36002"
export AGENT_MODEL_PATH="models/Qwen2.5-0.5B-Instruct"
export EXP_NAME="textworldexpress_benchmark_${SLURM_JOB_ID}"
export MODEL_SAVE_PATH="saves/${EXP_NAME}"
export TRAIN_FILE="AgentItemId/textworldexpress_benchmark_train.json"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
export ROLLOUT_N="${ROLLOUT_N:-2}"
export ROUNDS="${ROUNDS:-6}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-5}"
export TWX_STEP_LIMIT="${TWX_STEP_LIMIT:-50}"
export TWX_MAX_VALID_ACTIONS="${TWX_MAX_VALID_ACTIONS:-40}"

mkdir -p slurm_logs "${MODEL_SAVE_PATH}"

date
hostname
nvidia-smi
python --version
python -m pip show textworld-express py4j agentenv agentenv_textworldexpress | sed -n '1,220p'
java -version

textworldexpress --host 127.0.0.1 --port 36002 > "${MODEL_SAVE_PATH}/textworldexpress_server.log" 2>&1 &
TWX_PID=$!
trap 'kill ${TWX_PID} 2>/dev/null || true' EXIT

for i in $(seq 1 120); do
  if python - <<'PY'
import requests
requests.get("http://127.0.0.1:36002/", timeout=2).raise_for_status()
PY
  then
    break
  fi
  sleep 2
done

python - <<'PY'
import requests
base = "http://127.0.0.1:36002"
print(requests.get(base + "/", timeout=5).text)
env_id = requests.post(base + "/create", timeout=60).json()["id"]
reset = requests.post(base + "/reset", json={"id": env_id, "data_idx": 0}, timeout=60).json()
print(reset["observation"][:1000])
detail = requests.get(base + f"/detail?id={env_id}", timeout=30).json()
actions = detail.get("valid_actions", [])
action = actions[0] if actions else "look around"
print(requests.post(base + "/step", json={"id": env_id, "action": action}, timeout=60).json())
print(requests.post(base + "/close", json={"id": env_id}, timeout=30).json())
PY

bash examples/train/AgentGym-RL/textworldexpress_smoke_local.sh

date
