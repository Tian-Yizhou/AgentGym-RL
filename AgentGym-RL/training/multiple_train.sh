#!/bin/bash
#SBATCH --job-name=agrl_multi
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --constraint=sxm
#SBATCH --gres=gpu:2
#SBATCH --ntasks=12
#SBATCH --mem=240G
#SBATCH --output=output_%j.log
#SBATCH --error=error_%j.log

# Run multiple AgentGym training tasks in parallel inside one Slurm job.
# Each task is pinned to one GPU via CUDA_VISIBLE_DEVICES, and each task
# script launches its own env server on a different port (see port_map.txt)
# and traps EXIT so the server is torn down when its training finishes.
#
# To change which tasks run in parallel, edit the TASKS list below.
# The number of TASKS must match --gres=gpu:N (one GPU per task).

set -uo pipefail

# Resolve the repo root so the inner `cd AgentGym-RL` in each task script works
# regardless of where this job was submitted from.
SCRIPT_DIR="AgentGym-RL/training"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

TASKS=(sciworld textcraft)

if [ "${#TASKS[@]}" -lt 1 ]; then
  echo "No tasks specified in TASKS." >&2
  exit 1
fi

declare -a PIDS
declare -a TASK_NAMES
gpu_idx=0
for task in "${TASKS[@]}"; do
  script="${SCRIPT_DIR}/${task}_train.sh"
  if [ ! -f "${script}" ]; then
    echo "Missing task script: ${script}" >&2
    exit 1
  fi
  log_file="${REPO_ROOT}/output_${SLURM_JOB_ID:-local}_${task}.log"
  echo "Launching ${task} on GPU ${gpu_idx} (log: ${log_file})"
  CUDA_VISIBLE_DEVICES=${gpu_idx} bash "${script}" >"${log_file}" 2>&1 &
  PIDS+=($!)
  TASK_NAMES+=("${task}")
  gpu_idx=$((gpu_idx + 1))
done

# Wait for every child and collect exit codes.
overall=0
for i in "${!PIDS[@]}"; do
  wait "${PIDS[$i]}"
  rc=$?
  echo "${TASK_NAMES[$i]} (pid=${PIDS[$i]}) exited with code ${rc}"
  if [ "${rc}" -ne 0 ]; then
    overall=${rc}
  fi
done

exit ${overall}
