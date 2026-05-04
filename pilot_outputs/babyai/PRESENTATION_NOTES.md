# BabyAI Pilot Presentation Notes

## What is running

- BabyAI environment server runs as an HTTP service at `http://127.0.0.1:36001`.
- AgentGym `BabyAIEnvClient` successfully interacts with the server:
  - create environment
  - reset task
  - receive text observation and valid actions
  - send action
  - receive reward/done
  - close environment

## Current artifacts

- Reward/success curve: `pilot_outputs/babyai/reward_curve.png`
- Episode metrics: `pilot_outputs/babyai/metrics.csv`
- Full trajectories: `pilot_outputs/babyai/trajectories.jsonl`
- Case study: `pilot_outputs/babyai/case_study.md`
- RL run next steps: `pilot_outputs/babyai/RL_TRAINING_NEXT_STEPS.md`

## CPU smoke baseline result

This is not the final PPO/GRPO result. It is a CPU-only environment/client smoke test with simple policies.

| Policy | Episodes | Avg score | Success rate | Avg turns |
| --- | ---: | ---: | ---: | ---: |
| Random valid action | 12 | 0.387 | 41.7% | 5.42 |
| Goal-action heuristic | 12 | 0.610 | 66.7% | 3.58 |

## Case study finding

Successful case:

- Goal: `go to the red ball`
- Observation exposes valid macro action: `go to red ball 1`
- Heuristic thought maps the natural-language goal to that action.
- Result: success in 1 turn, score `0.8875`.

Failure case:

- Random policy picks valid but irrelevant actions, e.g. `move forward`, `pickup grey key 3`, `turn left`.
- It can get distracted by visible objects and lose access to the useful target-directed action.
- This is a useful motivation for RL/LLM agent training: action validity is not enough; the policy must preserve goal intent.

## Interesting pilot finding

The action interface is a major bottleneck:

- BabyAI goal text may say `go to the red ball`.
- Valid action schema may expose `go to red ball 1`.
- A naive string matcher failed because of the extra word `the`.
- Normalizing natural-language goals into executable action schema improved the smoke baseline.

This is a good point to discuss under "thinking process": the agent needs to bridge observation language, goal language, and environment action syntax.

## Full RL status

Prepared:

- Minimal training IDs: `AgentItemId/babyai_train.json`
- GPU smoke training script: `examples/train/AgentGym-RL/babyai_smoke_local.sh`

Blocked on current node:

- `nvidia-smi` cannot communicate with the NVIDIA driver.
- AgentGym-RL PPO/GRPO uses vLLM/FSDP, so it needs a GPU node.
- A local HuggingFace model directory is still needed, e.g. `models/Qwen2.5-0.5B-Instruct`.

Once on a GPU node, run:

```sh
conda activate agentgym-rl
babyai --host 127.0.0.1 --port 36001
```

Then:

```sh
conda activate agentgym-rl
AGENT_MODEL_PATH=models/Qwen2.5-0.5B-Instruct \
WANDB_MODE=offline \
bash examples/train/AgentGym-RL/babyai_smoke_local.sh
```

## Slide wording

> We completed environment setup and verified real agent-environment interaction for BabyAI. On the current CPU/login node we produced a smoke baseline, reward curve, and trajectory case study. The full PPO/GRPO loss curve is prepared but requires a GPU node plus a local HuggingFace model.
