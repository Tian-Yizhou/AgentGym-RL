#!/bin/bash
#SBATCH --partition=gengpu
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --constraint=sxm
#SBATCH --ntasks=6
#SBATCH --mem=120G
#SBATCH --job-name=webarena

# launch env server in background via the generic launcher
SCRIPT_DIR="AgentGym-RL/training"
SERVER_PORT=36001
bash "${SCRIPT_DIR}/start_env_server.sh" webshop "${SERVER_PORT}" &
SERVER_PID=$!
trap 'kill -TERM $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null' EXIT

# wait until env server accepts TCP on the port
echo "Waiting for webshop env server on 127.0.0.1:${SERVER_PORT} ..."
for _ in $(seq 1 60); do
  (echo > /dev/tcp/127.0.0.1/${SERVER_PORT}) >/dev/null 2>&1 && break
  sleep 2
done
(echo > /dev/tcp/127.0.0.1/${SERVER_PORT}) >/dev/null 2>&1 || { echo "webshop env server failed to start" >&2; exit 1; }
echo "webshop env server ready."

set -x
export VLLM_USE_MODELSCOPE=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=XFORMERS

task_name="webarena"

cd AgentGym-RL
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate agentgym-rl
export VLLM_ATTENTION_BACKEND=XFORMERS
export WANDB_BASE_URL=https://api.bandw.top

env_server_url="http://127.0.0.1:36001"

# start training
# wandb login 

pure_agent_model_name="Qwen2.5-7B-Instruct"
agent_model_path="models/${pure_agent_model_name}"

kl_coef=0.001
policy_learning_rate=1e-6
rollout_sample_num=4
train_batch_size=32
ppo_mini_batch_size=4
ppo_micro_batch_size_per_gpu=1
ppo_inner_epochs=2

total_epoches=25

model_save_dir="saves"
mkdir -p ${model_save_dir}
exp_name="webarena"
model_save_path=${model_save_dir}/${exp_name}

mkdir -p ${model_save_path}

HYDRA_FULL_ERROR=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True WANDB_MODE=online python3 -m verl.agent_trainer.main_ppo  \
    algorithm.adv_estimator=grpo \
    algorithm.rounds_ctrl.type=fixed \
    algorithm.rounds_ctrl.rounds=15 \
    data.train_file=AgentItemId/${task_name}_train.json \
    data.train_batch_size=${train_batch_size} \
    data.max_prompt_length=750 \
    data.max_response_length=14098 \
    actor_rollout_ref.agentgym.task_name=${task_name} \
    actor_rollout_ref.agentgym.env_addr=${env_server_url} \
    actor_rollout_ref.agentgym.timeout=600 \
    actor_rollout_ref.model.path=${agent_model_path} \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=${rollout_sample_num} \
    actor_rollout_ref.rollout.max_model_len=32768 \
    actor_rollout_ref.rollout.max_tokens=512 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.actor.ppo_epochs=${ppo_inner_epochs} \
    actor_rollout_ref.actor.optim.lr=${policy_learning_rate} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.rollout.rollout_log_dir=${model_save_path}/executer_logs \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    trainer.default_local_dir=${model_save_path} \
    trainer.project_name=xxx \
    trainer.experiment_name=${exp_name} \
    trainer.save_freq=25 \
    trainer.total_epochs=${total_epoches}
status=$?
exit $status