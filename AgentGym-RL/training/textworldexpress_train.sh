#!/bin/bash
#SBATCH --job-name=textworldexpress
#SBATCH --partition=gengpu
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --output=output_%j.log
#SBATCH --error=error_%j.log

# Slurm wrapper: start the TextWorldExpress env server, then run the cluster-agnostic
# training script (examples/train/<flavor>/textworldexpress_train.sh). Submit from the repo root:
#   sbatch AgentGym-RL/training/textworldexpress_train.sh
#   FLAVOR=ScalingInter-RL sbatch AgentGym-RL/training/textworldexpress_train.sh
# Any variable of the example script (MODEL_PATH, ROUNDS, TOTAL_TRAINING_STEPS, ...) can be
# exported before sbatch. Java comes from the agentenv-textworldexpress conda env (openjdk).

SCRIPT_DIR="AgentGym-RL/training"
SERVER_PORT="${SERVER_PORT:-36006}"
FLAVOR="${FLAVOR:-AgentGym-RL}"   # AgentGym-RL (fixed horizon) | ScalingInter-RL

bash "${SCRIPT_DIR}/start_env_server.sh" textworldexpress "${SERVER_PORT}" &
SERVER_PID=$!
trap 'kill -TERM $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null' EXIT

echo "Waiting for textworldexpress env server on 127.0.0.1:${SERVER_PORT} ..."
for _ in $(seq 1 60); do
  (echo > /dev/tcp/127.0.0.1/${SERVER_PORT}) >/dev/null 2>&1 && break
  sleep 2
done
(echo > /dev/tcp/127.0.0.1/${SERVER_PORT}) >/dev/null 2>&1 || { echo "textworldexpress env server failed to start" >&2; exit 1; }
echo "textworldexpress env server ready."

export ENV_SERVER_URL="http://127.0.0.1:${SERVER_PORT}"
bash "examples/train/${FLAVOR}/textworldexpress_train.sh"
