# BabyAI RL Training Next Steps

Current CPU/server smoke status:

- BabyAI environment server is running at `http://127.0.0.1:36001`.
- `BabyAIEnvClient` can create, reset, step, and close environments.
- CPU smoke artifacts are in `pilot_outputs/babyai/`.

Why the full RL run is not complete on this node:

- `nvidia-smi` fails with: `couldn't communicate with the NVIDIA driver`.
- `verl.agent_trainer.main_ppo` uses vLLM/FSDP and needs a visible GPU.
- The original script expects a local model at `models/Qwen2.5-7B-Instruct`; no local model directory was present.

Prepared files:

- Minimal BabyAI train IDs: `AgentItemId/babyai_train.json`
- Local GPU smoke script: `examples/train/AgentGym-RL/babyai_smoke_local.sh`

Run on a GPU node:

```sh
conda activate agentgym-rl
babyai --host 127.0.0.1 --port 36001
```

In another shell:

```sh
conda activate agentgym-rl
AGENT_MODEL_PATH=models/Qwen2.5-0.5B-Instruct \
WANDB_MODE=offline \
bash examples/train/AgentGym-RL/babyai_smoke_local.sh
```

Expected outputs after a successful RL smoke run:

- Console/W&B reward and loss metrics.
- Rollout trajectories in `saves/babyai_smoke/executer_logs`.
- Checkpoints or trainer outputs under `saves/babyai_smoke`.

Presentation-safe wording:

> We have completed the environment/client integration and a CPU interaction baseline. The full PPO/GRPO LLM training run is ready to launch, but this login node does not expose a working NVIDIA driver; it should be run on a GPU node with a local HuggingFace model path.
