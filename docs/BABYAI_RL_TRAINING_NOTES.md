# BabyAI RL Training Notes on Quest

This note records the current working setup for running AgentGym-RL with BabyAI on Northwestern Quest.

## Current Status

- BabyAI environment server can run on a compute node.
- `Qwen2.5-0.5B-Instruct` is downloaded locally at `models/Qwen2.5-0.5B-Instruct`.
- A smoke RL training job completed successfully:
  - Slurm job: `7094697`
  - State: `COMPLETED`
  - Exit code: `0:0`
  - Output directory: `saves/babyai_smoke_7094697`
- The smoke run used our tiny local data file `AgentItemId/babyai_train.json`.
- The official benchmark ID files are available at:
  - Train: `AgentGym-RL-Data-ID/train/babyai_train.json`
  - Eval: `AgentGym-RL-Data-ID/eval/babyai_test.json`

## Benchmark Data vs. Train From Scratch

The Hugging Face `AgentGym-RL-Data-ID` files are item ID lists, not pretrained checkpoints. They define which BabyAI tasks the RL trainer samples from.

So we can use the benchmark data directly for RL training, but this is still training the policy from the selected base model unless we also load an existing trained checkpoint.

In our current setup:

- Base model: `models/Qwen2.5-0.5B-Instruct`
- RL algorithm: GRPO through `verl.agent_trainer.main_ppo`
- Environment: BabyAI via AgentEnv server
- Training data: either the tiny smoke file or the official benchmark item IDs

The next natural step is to train on:

```bash
AgentGym-RL-Data-ID/train/babyai_train.json
```

instead of:

```bash
AgentItemId/babyai_train.json
```

## Scripts

Smoke training script:

```bash
examples/train/AgentGym-RL/babyai_smoke_local.sh
```

Slurm smoke job:

```bash
slurm/babyai_rl_smoke.sbatch
```

Slurm benchmark job:

```bash
slurm/babyai_rl_benchmark.sbatch
```

The local training script now accepts these environment variables:

```bash
TRAIN_FILE=AgentGym-RL-Data-ID/train/babyai_train.json
TRAIN_BATCH_SIZE=8
ROLLOUT_N=2
ROUNDS=8
TOTAL_EPOCHS=3
EXP_NAME=my_run
MODEL_SAVE_PATH=saves/my_run
AGENT_MODEL_PATH=models/Qwen2.5-0.5B-Instruct
ENV_SERVER_URL=http://127.0.0.1:36001
```

## Submit Benchmark Training

From the repo root:

```bash
sbatch slurm/babyai_rl_benchmark.sbatch
```

To override the default benchmark run length:

```bash
sbatch --export=ALL,TOTAL_EPOCHS=5,ROUNDS=8,TRAIN_BATCH_SIZE=8 slurm/babyai_rl_benchmark.sbatch
```

For the first non-smoke run, keep the settings conservative. After it completes, increase `TOTAL_EPOCHS`, then maybe `TRAIN_BATCH_SIZE` or `ROLLOUT_N`.

## Check Job Status

While the job is queued or running:

```bash
squeue -j <jobid>
```

If `squeue` only prints the header, the job is no longer running. Check its final state:

```bash
sacct -j <jobid> --format=JobID,JobName,State,ExitCode,Elapsed,NodeList
```

Successful jobs should show:

```text
COMPLETED  0:0
```

Check logs:

```bash
tail -n 100 slurm_logs/babyai_bench-<jobid>.err
tail -n 100 slurm_logs/babyai_bench-<jobid>.out
```

Follow logs live:

```bash
tail -f slurm_logs/babyai_bench-<jobid>.err
```

## Check Training Outputs

Each run writes to:

```bash
saves/<experiment_name>/
```

For example:

```bash
saves/babyai_benchmark_<jobid>/
```

Useful files:

```bash
saves/<experiment_name>/babyai_server.log
saves/<experiment_name>/executer_logs/step1/0.json
```

W&B offline logs are stored under:

```bash
AgentGym-RL/wandb/offline-run-*
```

Console logs include metrics such as:

```text
actor/kl_loss
actor/entropy_loss
actor/pg_loss
critic/score/mean
critic/rewards/mean
critic/task_round/mean
timing_s/step
```

These are enough to make the first loss and reward curves.

## Why We Needed Fixes

Quest has an older system `glibc`, so the available `flash-attn` wheel was not usable. The trainer originally assumed FlashAttention 2. We patched the code to allow non-FlashAttention execution and set:

```bash
VERL_ATTN_IMPLEMENTATION=sdpa
VLLM_ATTENTION_BACKEND=XFORMERS
```

Another failure happened because log-prob label indices were not `int64`. We patched `verl/utils/torch_functional.py` so labels are cast to `long()` before `gather`.

## Recommended Next Steps

1. Run `slurm/babyai_rl_benchmark.sbatch`.
2. Confirm `COMPLETED 0:0` with `sacct`.
3. Extract reward and loss metrics from W&B offline logs or Slurm console logs.
4. Visualize a few `executer_logs` examples to inspect the agent's reasoning and actions.
5. Increase training length after the first benchmark run works.
