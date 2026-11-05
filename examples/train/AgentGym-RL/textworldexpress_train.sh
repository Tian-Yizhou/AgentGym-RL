#!/bin/bash
# AgentGym-RL training on TextWorldExpress, fixed interaction horizon.
#
# Prerequisites (see AgentGym/agentenv-textworldexpress/README.md):
#   1. env server running:   textworldexpress --host 127.0.0.1 --port 36005     (conda env agentenv-textworldexpress)
#   2. item ids generated:    python AgentGym/agentenv-textworldexpress/scripts/make_data.py --out-dir AgentGym-RL
#   3. run from the repo root with the agentgym-rl conda env active (or let this script activate it).
#
# Every setting below can be overridden from the environment, e.g.
#   N_GPUS=4 ROUNDS=20 MODEL_PATH=models/Qwen2.5-3B-Instruct bash examples/train/AgentGym-RL/textworldexpress_train.sh
#
# Token budget (measured with the Qwen2.5 tokenizer on TWX default settings): a turn costs
# ~100-350 tokens of observation + action list, plus the model's reply (<= MAX_TOKENS). Over
# ROUNDS turns that must fit MAX_RESPONSE_LENGTH, otherwise the trajectory is truncated for
# training while the reward still reflects the whole episode. CookingWorld (the longest game)
# needs ~340 tokens/turn; the other games 100-200.
set -x

if [ "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV:-agentgym-rl}" ]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV:-agentgym-rl}"
fi

export VLLM_USE_MODELSCOPE=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=XFORMERS
export WANDB_MODE="${WANDB_MODE:-offline}"

task_name="textworldexpress"
env_server_url="${ENV_SERVER_URL:-http://127.0.0.1:36005}"
agent_model_path="${MODEL_PATH:-models/Qwen2.5-7B-Instruct}"   # relative to AgentGym-RL/
n_gpus="${N_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}"

kl_coef="${KL_COEF:-0.001}"
policy_learning_rate="${LR:-1e-6}"
rollout_sample_num="${ROLLOUT_N:-8}"
train_batch_size="${TRAIN_BATCH_SIZE:-16}"
ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE:-8}"
ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
ppo_inner_epochs="${PPO_EPOCHS:-1}"
total_epoches="${TOTAL_EPOCHS:-3}"
total_training_steps="${TOTAL_TRAINING_STEPS:-null}"   # set e.g. 2 for a smoke test
rounds="${ROUNDS:-30}"
max_response_length="${MAX_RESPONSE_LENGTH:-12288}"
max_tokens="${MAX_TOKENS:-200}"
save_freq="${SAVE_FREQ:-25}"

exp_name="${EXP_NAME:-textworldexpress_fixed${rounds}}"
model_save_path="${SAVE_DIR:-saves}/${exp_name}"

cd AgentGym-RL || exit 1
mkdir -p "${model_save_path}"

HYDRA_FULL_ERROR=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 -m verl.agent_trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.rounds_ctrl.type=fixed \
    algorithm.rounds_ctrl.rounds=${rounds} \
    data.train_file=AgentItemId/train/${task_name}_train.json \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=1024 \
    data.max_response_length=${max_response_length} \
    actor_rollout_ref.agentgym.task_name=${task_name} \
    actor_rollout_ref.agentgym.env_addr=${env_server_url} \
    actor_rollout_ref.agentgym.timeout=600 \
    actor_rollout_ref.model.path=${agent_model_path} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=${kl_coef} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=${rollout_sample_num} \
    actor_rollout_ref.rollout.max_model_len=32768 \
    actor_rollout_ref.rollout.max_tokens=${max_tokens} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.actor.ppo_epochs=${ppo_inner_epochs} \
    actor_rollout_ref.actor.optim.lr=${policy_learning_rate} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.rollout.rollout_log_dir=${model_save_path}/executer_logs \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    trainer.n_gpus_per_node=${n_gpus} \
    trainer.default_local_dir=${model_save_path} \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${WANDB_PROJECT:-agentgym-rl} \
    trainer.experiment_name=${exp_name} \
    trainer.save_freq=${save_freq} \
    trainer.total_epochs=${total_epoches} \
    trainer.total_training_steps=${total_training_steps}
status=$?
exit $status
