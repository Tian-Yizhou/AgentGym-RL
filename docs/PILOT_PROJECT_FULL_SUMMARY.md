# RL Training Framework for LLM Agents: Project Summary

This document summarizes the project work, implementation decisions, commands, scripts, artifacts, and current status.

## 0. Current Status

Completed:

- BabyAI environment runs through AgentGym/AgentEnv.
- BabyAI RL smoke run completed.
- BabyAI benchmark-scale RL run completed with about 302 parsed training steps.
- TextWorldExpress was added as a new environment.
- TextWorldExpress smoke run completed.
- TextWorldExpress baseline was improved with valid-action canonicalization.
- Training metrics, curves, and case studies are generated under `pilot_outputs/final_artifacts/`.

Currently running:

```bash
squeue -j 7111047
```

Job:

```text
7111047 twx_bench
```

Purpose:

- Long TextWorldExpress benchmark-scale run.
- 256 item IDs, batch size 4, 5 epochs.
- Expected training steps: about `256 / 4 * 5 = 320`.

Progress at last check:

```text
Elapsed: about 50 minutes
Latest rollout output: step113
Estimated total time: about 2.4 hours
Estimated remaining: about 1.5 hours
```

Check progress:

```bash
squeue -j 7111047 -o '%i %P %j %u %t %M %l %D %R'
find saves/textworldexpress_benchmark_7111047/executer_logs -maxdepth 1 -type d -name 'step*' | wc -l
grep -n "step:" slurm_logs/twx_bench-7111047.out | tail
tail -f slurm_logs/twx_bench-7111047.err
```

## 1. Project Goal

The assignment goal is:

```text
Set up an RL training framework for LLM agents.
```

Requirements:

1. Set up the training framework and make it run successfully.
2. Show training curves: loss and reward.
3. Show case studies, especially thinking process.

Bonus:

- Run more environments/baselines.
- Improve a baseline.
- Add a new environment to the existing repo.
- Report interesting RL findings.

Our mapping:

| Requirement | What we did |
|---|---|
| Framework running | AgentGym-RL + Verl PPO/GRPO + Qwen2.5-0.5B-Instruct + AgentEnv servers |
| Training curve | Parsed Slurm/W&B-style logs into CSV and PNG curves |
| Case study | Extracted rollout conversations from `executer_logs` |
| More envs | BabyAI and TextWorldExpress |
| New env | Added `AgentGym/agentenv-textworldexpress` |
| Baseline improvement | TextWorldExpress valid-action canonicalization |
| Interesting finding | LLMs often fail action grounding even when valid actions are listed |

## 2. Repository Layout

Important paths:

```text
AgentGym/
  agentenv/
  agentenv-babyai/
  agentenv-textworldexpress/

AgentGym-RL/
  verl/

AgentItemId/
  babyai_train.json
  textworldexpress_train.json
  textworldexpress_benchmark_train.json

slurm/
  babyai_rl_smoke.sbatch
  babyai_rl_benchmark.sbatch
  textworldexpress_rl_smoke.sbatch
  textworldexpress_rl_benchmark.sbatch

examples/train/AgentGym-RL/
  babyai_smoke_local.sh
  textworldexpress_smoke_local.sh

scripts/
  build_pilot_artifacts.py

pilot_outputs/final_artifacts/
  metrics CSVs
  curve PNGs
  case_studies.md
  completion_report.md
```

## 3. Conceptual Architecture

The training stack has three layers:

```text
LLM policy trainer
  -> AgentGym client
    -> AgentEnv HTTP server
      -> actual environment backend
```

For BabyAI:

```text
verl.agent_trainer.main_ppo
  -> BabyAIEnvClient
    -> babyai FastAPI server
      -> BabyAI environment
```

For TextWorldExpress:

```text
verl.agent_trainer.main_ppo
  -> TextWorldExpressEnvClient
    -> textworldexpress FastAPI server
      -> textworld_express Python package
        -> Java/Py4J backend
```

Why two environment setups exist:

- The agent/trainer environment contains model/training dependencies: PyTorch, vLLM, Verl, Transformers.
- The task environment contains backend-specific dependencies: BabyAI, TextWorldExpress, Java/Py4J, etc.
- AgentGym decouples them through HTTP so the trainer can interact with many environment types using one protocol.

## 4. Base Model

Base model:

```text
models/Qwen2.5-0.5B-Instruct
```

This was used for all current runs.

The model is small enough to run on one Quest GPU with vLLM/FSDP in this repo.

## 5. Quest / Slurm Setup

General Slurm pattern:

```bash
sbatch slurm/<script>.sbatch
squeue -j <jobid>
sacct -j <jobid> --format=JobID,JobName,State,ExitCode,Elapsed,NodeList
tail -f slurm_logs/<job-name>-<jobid>.err
tail -f slurm_logs/<job-name>-<jobid>.out
```

GPU jobs use:

```bash
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
```

For TextWorldExpress, Java is required:

```bash
module load 'java/jdk-17.0.2+8'
java -version
```

## 6. BabyAI Work

### 6.1 BabyAI Server

BabyAI server package:

```text
AgentGym/agentenv-babyai/
```

Server entry:

```bash
babyai --host 127.0.0.1 --port 36001
```

Health check:

```bash
python - <<'PY'
import requests
print(requests.get("http://127.0.0.1:36001/", timeout=5).text)
PY
```

Expected response:

```text
This is environment BabyAI.
```

### 6.2 BabyAI Smoke Script

Script:

```text
examples/train/AgentGym-RL/babyai_smoke_local.sh
```

Important configurable variables:

```bash
ENV_SERVER_URL=http://127.0.0.1:36001
AGENT_MODEL_PATH=models/Qwen2.5-0.5B-Instruct
TRAIN_FILE=AgentItemId/babyai_train.json
TRAIN_BATCH_SIZE=8
ROLLOUT_N=2
ROUNDS=8
TOTAL_EPOCHS=1
```

Core command:

```bash
python3 -m verl.agent_trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.rounds_ctrl.type=fixed \
  algorithm.rounds_ctrl.rounds=${rounds} \
  data.train_file=../${train_file} \
  data.train_batch_size=${train_batch_size} \
  actor_rollout_ref.agentgym.task_name=babyai \
  actor_rollout_ref.agentgym.env_addr=${env_server_url} \
  actor_rollout_ref.model.path=../${agent_model_path} \
  actor_rollout_ref.rollout.n=${rollout_n} \
  trainer.total_epochs=${total_epochs}
```

### 6.3 BabyAI Slurm Scripts

Smoke:

```bash
sbatch slurm/babyai_rl_smoke.sbatch
```

Successful smoke job:

```text
7094697
```

Benchmark:

```bash
sbatch slurm/babyai_rl_benchmark.sbatch
```

Successful benchmark job:

```text
7099097
```

Benchmark data:

```text
AgentGym-RL-Data-ID/train/babyai_train.json
AgentGym-RL-Data-ID/eval/babyai_test.json
```

### 6.4 BabyAI Issues Fixed

FlashAttention issue:

- Quest system `glibc` was too old for the available `flash-attn` wheel.
- We patched Verl worker code to allow fallback when `flash_attn` cannot import.
- We set:

```bash
VERL_ATTN_IMPLEMENTATION=sdpa
VLLM_ATTENTION_BACKEND=XFORMERS
```

Dtype issue:

- Training failed once with:

```text
RuntimeError: gather(): Expected dtype int64 for index
```

- Fixed in:

```text
AgentGym-RL/verl/utils/torch_functional.py
```

by casting labels to long before `gather`.

## 7. TextWorldExpress Work

### 7.1 Why TextWorldExpress

TextWorldExpress is a text-game environment. It is useful because:

- It is lightweight compared with browser environments.
- It exposes valid actions explicitly.
- It tests whether an LLM can ground free-form reasoning into exact executable actions.

### 7.2 Dependency Setup

Installed:

```bash
conda activate agentgym-rl
python -m pip install textworld-express py4j
python -m pip install -e AgentGym/agentenv
python -m pip install -e AgentGym/agentenv-textworldexpress
```

Java:

```bash
module load 'java/jdk-17.0.2+8'
```

Minimal API test:

```bash
module load 'java/jdk-17.0.2+8'
conda run -n agentgym-rl python -c "from textworld_express import TextWorldExpressEnv; env=TextWorldExpressEnv(envStepLimit=20); env.load(gameName='twc', gameParams='numLocations=3,includeDoors=1'); obs,infos=env.reset(seed=0, gameFold='train', generateGoldPath=True); print(obs[:200]); print(len(infos.get('validActions', []))); env.close()"
```

### 7.3 New Environment Package

Added:

```text
AgentGym/agentenv-textworldexpress/
```

Main files:

```text
pyproject.toml
agentenv_textworldexpress/__init__.py
agentenv_textworldexpress/environment.py
agentenv_textworldexpress/server.py
agentenv_textworldexpress/model.py
agentenv_textworldexpress/launch.py
```

Entry command:

```bash
textworldexpress --host 127.0.0.1 --port 36002
```

HTTP endpoints:

```text
GET  /
POST /create
POST /reset
POST /step
GET  /observation
GET  /detail
POST /close
```

### 7.4 TextWorldExpress Server Design

The server manager is in:

```text
AgentGym/agentenv-textworldexpress/agentenv_textworldexpress/environment.py
```

It maintains:

```python
self.env: dict[int, TextWorldExpressEnv]
self.info: dict[int, dict]
self.games = self._build_games()
```

Game set:

```text
200 TWC seeds
60 coin seeds
260 total indexed tasks
```

Each `reset(id, data_idx)` maps `data_idx` to:

```python
game = self.games[data_idx % len(self.games)]
```

This makes item IDs stable:

```text
textworldexpress_0
textworldexpress_1
...
```

### 7.5 TextWorldExpress Client Design

Added:

```text
AgentGym/agentenv/agentenv/envs/textworldexpress.py
```

Registered in:

```text
AgentGym/agentenv/agentenv/envs/__init__.py
AgentGym-RL/verl/utils/agentgym/client.py
```

Registry key:

```python
"textworldexpress": TextWorldExpressEnvClient
```

This matters because Verl parses:

```python
task_name = item_id.split("_")[0]
item_id = int(item_id.split("_")[-1])
```

So item IDs must be:

```json
{"item_id": "textworldexpress_0"}
```

### 7.6 Baseline Improvement

Problem found:

- The model often produced plausible but invalid actions.
- Example:

```text
Action: put clean blue bra in wardrobe
```

when valid actions only included:

```text
put clean blue bra in TV stand
put clean blue bra in sofa
...
```

Improvement:

- Cache `valid_actions`.
- If generated action matches valid action ignoring case, canonicalize it.
- If no `Action:` line exists but the response mentions exactly one valid action, extract that action.

Verification:

```text
Input: I think I should take clean blue bra now.
Parsed/executed: take clean blue bra
Reward: 0.125
```

Observed improvement:

```text
Unknown action count before: 54
Unknown action count after: 47
```

This is modest, but useful as a finding.

## 8. TextWorldExpress Runs

Smoke:

```bash
sbatch slurm/textworldexpress_rl_smoke.sbatch
```

Successful jobs:

```text
7103978
7107157
```

Benchmark-scale run:

```bash
sbatch slurm/textworldexpress_rl_benchmark.sbatch
```

Current job:

```text
7111047
```

Configuration:

```bash
TRAIN_FILE=AgentItemId/textworldexpress_benchmark_train.json
TRAIN_BATCH_SIZE=4
TOTAL_EPOCHS=5
ROUNDS=6
ROLLOUT_N=2
```

Expected steps:

```text
256 / 4 * 5 = 320
```

## 9. Artifact Generation

Script:

```bash
conda run -n agentgym-rl python scripts/build_pilot_artifacts.py
```

Output:

```text
pilot_outputs/final_artifacts/
```

Current files:

```text
babyai_benchmark_metrics.csv
babyai_smoke_metrics.csv
textworldexpress_metrics.csv
babyai_smoke_training_curves.png
babyai_benchmark_training_curves.png
textworldexpress_training_curves.png
smoke_comparison_training_curves.png
case_studies.md
completion_report.md
```

Important note:

- The current TextWorldExpress plot is still from the short smoke run.
- After `7111047` finishes, update `scripts/build_pilot_artifacts.py` to parse:

```text
slurm_logs/twx_bench-7111047.out
saves/textworldexpress_benchmark_7111047/
```

Then rerun:

```bash
conda run -n agentgym-rl python scripts/build_pilot_artifacts.py
```

## 10. Case Study Findings

### BabyAI

BabyAI trajectories show the model can often reason about:

- object color,
- doors,
- keys,
- navigation,
- whether to pick up or go through objects.

But it also sometimes:

- loops,
- over-explains,
- chooses suboptimal navigation,
- fails when the state changes after several turns.

Useful artifact:

```text
pilot_outputs/final_artifacts/case_studies.md
```

### TextWorldExpress

Main finding:

```text
LLM reasoning can be linguistically plausible but not action-grounded.
```

Example failure pattern:

1. The model sees valid actions.
2. It reasons about a plausible object/container.
3. It generates an action not in the valid list.
4. The environment returns `Unknown action`.

This is important because it shows that RL for LLM agents is not only about reasoning quality. It also requires grounding into the environment action space.

## 11. Presentation Talking Points

Minimal story:

1. We set up AgentGym-RL on Quest with Qwen2.5-0.5B-Instruct.
2. We ran BabyAI end-to-end and produced benchmark-scale training curves.
3. We added TextWorldExpress as a new environment.
4. We ran TextWorldExpress through the same RL framework.
5. We found action grounding is a major failure mode.
6. We added a small action canonicalization baseline improvement.
7. We generated curves, CSVs, and case studies.

Core technical contribution:

```text
One RL training framework, multiple LLM-agent environments, one newly integrated environment, and a concrete baseline improvement.
```

## 12. Commands Cheat Sheet

Check jobs:

```bash
squeue -u rvt4270
squeue -j <jobid>
sacct -j <jobid> --format=JobID,JobName,State,ExitCode,Elapsed,NodeList
```

Run BabyAI benchmark:

```bash
sbatch slurm/babyai_rl_benchmark.sbatch
```

Run TextWorldExpress smoke:

```bash
sbatch slurm/textworldexpress_rl_smoke.sbatch
```

Run TextWorldExpress benchmark:

```bash
sbatch slurm/textworldexpress_rl_benchmark.sbatch
```

Tail logs:

```bash
tail -f slurm_logs/babyai_bench-7099097.out
tail -f slurm_logs/twx_bench-7111047.err
tail -f slurm_logs/twx_bench-7111047.out
```

Regenerate final artifacts:

```bash
conda run -n agentgym-rl python scripts/build_pilot_artifacts.py
```

Count TextWorldExpress progress:

```bash
find saves/textworldexpress_benchmark_7111047/executer_logs -maxdepth 1 -type d -name 'step*' | wc -l
```

## 13. What To Do After `7111047` Finishes

1. Confirm job completion:

```bash
squeue -j 7111047
sacct -j 7111047 --format=JobID,JobName,State,ExitCode,Elapsed,NodeList
```

2. Count metrics:

```bash
grep -n "step:" slurm_logs/twx_bench-7111047.out | wc -l
```

3. Update artifact script to use `twx_bench-7111047.out`.

4. Regenerate plots:

```bash
conda run -n agentgym-rl python scripts/build_pilot_artifacts.py
```

5. Check final plots:

```bash
ls -lh pilot_outputs/final_artifacts/*curves.png
```

