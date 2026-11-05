#!/bin/bash
# Generic AgentGym env-server launcher.
#
# Usage: start_env_server.sh <env_name> <port> [host]
#   env_name: textcraft | sciworld | alfworld | webshop | webarena | babyai | searchqa | textworldexpress
#   port:     TCP port to bind (e.g. 36003)
#   host:     bind address (default 127.0.0.1)
#
# Activates the right conda env, cd's into the env package directory (some
# servers load data files via relative paths), then exec's the server CLI.
# Because we exec, the PID of `bash start_env_server.sh ... &` is the server
# itself, so the caller's `kill -TERM $SERVER_PID` reaches the real process.

set -euo pipefail

ENV_NAME="${1:?env_name required: textcraft|sciworld|alfworld|webshop|webarena|babyai|searchqa|textworldexpress}"
PORT="${2:?port required}"
HOST="${3:-127.0.0.1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# AgentGym-RL/AgentGym-RL/training/.. -> AgentGym-RL/AgentGym-RL
# AgentGym-RL/AgentGym-RL/.. -> AgentGym-RL (repo root)
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
AGENTGYM_ROOT="${REPO_ROOT}/AgentGym"

case "${ENV_NAME}" in
  textcraft)
    CONDA_ENV="agentenv-textcraft"
    WORK_DIR="${AGENTGYM_ROOT}/agentenv-textcraft"
    BIN="textcraft"
    ;;
  sciworld)
    CONDA_ENV="agentenv-sciworld"
    WORK_DIR="${AGENTGYM_ROOT}/agentenv-sciworld"
    BIN="sciworld"
    ;;
  alfworld)
    CONDA_ENV="agentenv-alfworld"
    WORK_DIR="${AGENTGYM_ROOT}/agentenv-alfworld"
    BIN="alfworld"
    ;;
  webshop)
    CONDA_ENV="agentenv-webshop"
    WORK_DIR="${AGENTGYM_ROOT}/agentenv-webshop"
    BIN="webshop"
    ;;
  webarena)
    CONDA_ENV="agentenv-webarena"
    WORK_DIR="${AGENTGYM_ROOT}/agentenv-webarena"
    BIN="webarena"
    ;;
  babyai)
    CONDA_ENV="agentenv-babyai"
    WORK_DIR="${AGENTGYM_ROOT}/agentenv-babyai"
    BIN="babyai"
    ;;
  searchqa)
    CONDA_ENV="agentenv-searchqa"
    WORK_DIR="${AGENTGYM_ROOT}/agentenv-searchqa"
    BIN="searchqa"
    ;;
  textworldexpress)
    CONDA_ENV="agentenv-textworldexpress"
    WORK_DIR="${AGENTGYM_ROOT}/agentenv-textworldexpress"
    BIN="textworldexpress"
    ;;
  *)
    echo "Unknown env: ${ENV_NAME}" >&2
    echo "Supported: textcraft sciworld alfworld webshop webarena babyai searchqa textworldexpress" >&2
    exit 2
    ;;
esac

if [ ! -d "${WORK_DIR}" ]; then
  echo "Env package directory not found: ${WORK_DIR}" >&2
  exit 1
fi

# Use conda's shell hook instead of `source activate` — the legacy
# `activate` script reads $1/$2 from the parent shell, which here are
# the env_name/port positional args, causing it to abort with
# "ArgumentError: activate does not accept more than one argument".
CONDA_BASE="$(conda info --base 2>/dev/null)"
if [ -z "${CONDA_BASE}" ] || [ ! -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]; then
  echo "Cannot locate conda installation (conda info --base failed)." >&2
  exit 1
fi
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
cd "${WORK_DIR}"

echo "[start_env_server] ${ENV_NAME} -> conda=${CONDA_ENV} dir=${WORK_DIR} bind=${HOST}:${PORT}"
exec "${BIN}" --host "${HOST}" --port "${PORT}"
