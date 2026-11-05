#!/bin/bash
# Evaluate a model on TextWorldExpress (test fold; per-game breakdown printed at the end).
#
#   CKPT_PATH=saves/<exp>/global_step_150/actor bash examples/eval/textworldexpress_eval.sh   # trained checkpoint
#   MODEL_PATH=models/Qwen2.5-7B-Instruct      bash examples/eval/textworldexpress_eval.sh   # any HF model (baseline)
#   EVAL_SET=textworldexpress_dev ...                                                       # dev fold instead
#
# Paths are relative to AgentGym-RL/. Needs a running env server (see the training script).
# "Avg" in the output is the mean TWX score (partial credit counts, failures can be negative);
# "Pass" counts any score > 0, so it is NOT the success rate -- success means score == 1.0
# (see the per-item rewards in ${LOG_DIR}).
set -x

if [ "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV:-agentgym-rl}" ]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV:-agentgym-rl}"
fi

export VLLM_USE_MODELSCOPE=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=XFORMERS

task_name="textworldexpress"
env_server_url="${ENV_SERVER_URL:-http://127.0.0.1:36005}"
eval_set="${EVAL_SET:-textworldexpress}"   # directory under AgentEval/
n_gpus="${N_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
sample_num="${SAMPLE_NUM:-1}"
max_rounds="${MAX_ROUNDS:-30}"
log_dir="${LOG_DIR:-eval_logs/${eval_set}}"

cd AgentGym-RL || exit 1

if [ -n "${CKPT_PATH:-}" ]; then
  python scripts/model_merger.py --local_dir "${CKPT_PATH}" || exit 1
  model_path="${CKPT_PATH}/huggingface"
else
  model_path="${MODEL_PATH:-models/Qwen2.5-7B-Instruct}"
fi

HYDRA_FULL_ERROR=1 python3 -m verl.agent_trainer.main_generation \
    data.path=AgentEval/${eval_set} \
    data.max_prompt_length=1024 \
    data.max_response_length=${MAX_RESPONSE_LENGTH:-12288} \
    data.n_samples=${sample_num} \
    data.batch_size=${BATCH_SIZE:-32} \
    agentgym.task_name=${task_name} \
    agentgym.env_addr=${env_server_url} \
    agentgym.max_rounds=${max_rounds} \
    agentgym.timeout=500 \
    model.path=${model_path} \
    trainer.n_gpus_per_node=${n_gpus} \
    rollout.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION:-0.9} \
    rollout.temperature=${TEMPERATURE:-1} \
    rollout.max_model_len=32768 \
    rollout.max_tokens=${MAX_TOKENS:-200} \
    rollout.tensor_model_parallel_size=1 \
    rollout.rollout_log_dir=${log_dir}
status=$?
exit $status
