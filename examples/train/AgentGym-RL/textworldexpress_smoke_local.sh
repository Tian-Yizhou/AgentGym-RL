set -euo pipefail
set -x

export VLLM_USE_MODELSCOPE="${VLLM_USE_MODELSCOPE:-0}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if ! nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: No working NVIDIA driver/GPU is visible. AgentGym-RL vLLM training needs a GPU node." >&2
  exit 2
fi

task_name="textworldexpress"
env_server_url="${ENV_SERVER_URL:-http://127.0.0.1:36002}"
agent_model_path="${AGENT_MODEL_PATH:-models/Qwen2.5-0.5B-Instruct}"
exp_name="${EXP_NAME:-textworldexpress_smoke}"
model_save_path="${MODEL_SAVE_PATH:-saves/${exp_name}}"
train_file="${TRAIN_FILE:-AgentItemId/${task_name}_train.json}"
train_batch_size="${TRAIN_BATCH_SIZE:-4}"
rollout_n="${ROLLOUT_N:-2}"
rounds="${ROUNDS:-6}"
total_epochs="${TOTAL_EPOCHS:-1}"

if [ ! -e "${agent_model_path}" ]; then
  echo "ERROR: model path not found: ${agent_model_path}" >&2
  echo "Set AGENT_MODEL_PATH to a local HuggingFace model directory before running." >&2
  exit 3
fi

mkdir -p "${model_save_path}"

cd AgentGym-RL

python3 -m verl.agent_trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.rounds_ctrl.type=fixed \
  algorithm.rounds_ctrl.rounds=${rounds} \
  data.train_file=../${train_file} \
  data.train_batch_size=${train_batch_size} \
  data.max_prompt_length=1536 \
  data.max_response_length=2048 \
  actor_rollout_ref.agentgym.task_name=${task_name} \
  actor_rollout_ref.agentgym.env_addr=${env_server_url} \
  actor_rollout_ref.agentgym.timeout=300 \
  actor_rollout_ref.model.path=../${agent_model_path} \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
  actor_rollout_ref.rollout.n=${rollout_n} \
  actor_rollout_ref.rollout.max_model_len=8192 \
  actor_rollout_ref.rollout.max_tokens=128 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.rollout_log_dir=../${model_save_path}/executer_logs \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  trainer.n_gpus_per_node=1 \
  trainer.logger=['console','wandb'] \
  trainer.default_local_dir=../${model_save_path} \
  trainer.project_name=agentgym_rl_pilot \
  trainer.experiment_name=${exp_name} \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.total_epochs=${total_epochs}

