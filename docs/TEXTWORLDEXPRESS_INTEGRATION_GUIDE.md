# TextWorldExpress Integration Guide for AgentGym-RL

This guide describes how to add TextWorldExpress as a new AgentGym environment and then run it through the existing AgentGym-RL trainer.

Useful upstream references:

- TextWorldExpress GitHub: https://github.com/cognitiveailab/TextWorldExpress
- TextWorldExpress PyPI: https://pypi.org/project/textworld-express/

## Goal

Add a new environment as a bonus item:

```text
TextWorldExpress -> AgentEnv HTTP server -> AgentGym client -> AgentGym-RL rollout/training
```

The safest internal task name is:

```text
textworldexpress
```

Then item IDs should look like:

```json
[
  {"item_id": "textworldexpress_0"},
  {"item_id": "textworldexpress_1"}
]
```

This matters because the rollout code parses task name and numeric ID from `item_id`:

```python
task_name = item_id.split("_")[0]
item_id = int(item_id.split("_")[-1])
```

## High-Level Plan

1. Verify `textworld_express` can run in `agentgym-rl`.
2. Create a new package: `AgentGym/agentenv-textworldexpress`.
3. Implement the standard AgentEnv HTTP endpoints:
   - `GET /`
   - `POST /create`
   - `POST /reset`
   - `POST /step`
   - `GET /observation`
   - `POST /close`
4. Add a new `TextWorldExpressEnvClient` in `AgentGym/agentenv/agentenv/envs/`.
5. Register the client in:
   - `AgentGym/agentenv/agentenv/envs/__init__.py`
   - `AgentGym-RL/verl/utils/agentgym/client.py`
6. Create train/eval item ID files.
7. Add a smoke test script and then a Slurm training script.

## Step 1: Install and Verify TextWorldExpress

Current status in `agentgym-rl`:

```bash
conda run -n agentgym-rl python -c "import importlib.util; print(importlib.util.find_spec('textworld_express'))"
```

Current result:

```text
None
```

So it still needs to be installed.

Try:

```bash
conda activate agentgym-rl
python -m pip install textworld-express py4j
```

TextWorldExpress uses a Java/Scala backend through `py4j`, so also check Java:

```bash
java -version
```

If Java is missing on Quest, inspect/load a Java module:

```bash
module spider java
module load java
java -version
```

Minimal local Python smoke test:

```python
from textworld_express import TextWorldExpressEnv

env = TextWorldExpressEnv(envStepLimit=50)
env.load(gameName="twc", gameParams="numLocations=3,includeDoors=1")

obs, infos = env.reset(seed=0, gameFold="train", generateGoldPath=True)
print(obs)
print(sorted(infos["validActions"])[:20])

action = sorted(infos["validActions"])[0]
obs, reward, done, infos = env.step(action)
print(action, reward, done)
print(obs)

env.shutdown()
```

TextWorldExpress supports at least these game names:

```text
twc
cookingworld
coin
```

Start with `twc` because it is a straightforward text-world commonsense environment.

## Step 2: New Server Package Layout

Create:

```text
AgentGym/agentenv-textworldexpress/
  pyproject.toml
  README.md
  agentenv_textworldexpress/
    __init__.py
    environment.py
    launch.py
    model.py
    server.py
```

Use these existing environments as templates:

```text
AgentGym/agentenv-babyai/
AgentGym/agentenv-sciworld/
AgentGym/agentenv-textcraft/
```

## Step 3: `pyproject.toml`

Draft:

```toml
[project]
name = "agentenv_textworldexpress"
version = "0.0.1"
description = "AgentEnv wrapper for TextWorldExpress"
dependencies = [
    "fastapi",
    "uvicorn",
    "textworld-express",
    "py4j"
]
requires-python = "~=3.10"
readme = "README.md"
license = {text = "MIT"}

[project.scripts]
textworldexpress = "agentenv_textworldexpress:launch"
```

## Step 4: Server Schema

`agentenv_textworldexpress/model.py`:

```python
from pydantic import BaseModel


class StepRequestBody(BaseModel):
    id: int
    action: str


class ResetRequestBody(BaseModel):
    id: int
    data_idx: int


class CloseRequestBody(BaseModel):
    id: int
```

## Step 5: Environment Manager

`agentenv_textworldexpress/environment.py` should manage many TextWorldExpress env instances.

Important detail: the installed Quest version is `textworld-express==1.1.0`. Its constructor signature is:

```python
TextWorldExpressEnv(serverPath=None, envStepLimit=100)
```

Older examples mention `threadNum`, but this version does not accept it. Py4J uses dynamic ports internally.

Skeleton:

```python
import threading
from textworld_express import TextWorldExpressEnv


class TextWorldExpressServer:
    def __init__(self):
        self._max_id = 0
        self.env = {}
        self.info = {}
        self._lock = threading.Lock()
        self.games = self._build_games()

    def _build_games(self):
        games = []
        for seed in range(200):
            games.append({
                "gameName": "twc",
                "gameParams": "numLocations=3,includeDoors=1",
                "gameFold": "train",
                "seed": seed,
            })
        return games

    def create(self):
        with self._lock:
            idx = self._max_id
            self._max_id += 1

            env = TextWorldExpressEnv(envStepLimit=50)
        self.env[idx] = env
        self.info[idx] = {"deleted": False, "done": False}
        return {"id": idx}

    def reset(self, idx: int, data_idx: int):
        self._check_id(idx, is_reset=True)
        game = self.games[data_idx % len(self.games)]

        env = self.env[idx]
        env.load(gameName=game["gameName"], gameParams=game["gameParams"])
        obs, infos = env.reset(
            seed=game["seed"],
            gameFold=game["gameFold"],
            generateGoldPath=True,
        )

        payload = {
            "observation": self._format_observation(obs, infos),
            "reward": 0.0,
            "score": 0.0,
            "done": False,
            "valid_actions": sorted(infos.get("validActions", [])),
        }
        self.info[idx].update(payload)
        return payload

    def step(self, idx: int, action: str):
        self._check_id(idx)
        obs, score, done, infos = self.env[idx].step(action)

        payload = {
            "observation": self._format_observation(obs, infos),
            "reward": float(score),
            "score": float(score),
            "done": bool(done),
            "valid_actions": sorted(infos.get("validActions", [])),
        }
        self.info[idx].update(payload)
        return payload

    def observe(self, idx: int):
        self._check_id(idx)
        return self.info[idx]["observation"]

    def close(self, idx: int):
        self._check_id(idx)
        self.env[idx].close()
        self.info[idx]["deleted"] = True
        return True

    def _format_observation(self, obs: str, infos: dict):
        valid = sorted(infos.get("validActions", []))[:30]
        valid_text = "\n".join(f"- {a}" for a in valid)
        return f"{obs}\n\nValid actions:\n{valid_text}"

    def _check_id(self, idx: int, is_reset: bool = False):
        if idx not in self.info:
            raise ValueError(f"Invalid environment id: {idx}")
        if self.info[idx].get("deleted"):
            raise ValueError(f"Environment {idx} has been deleted.")
        if not is_reset and self.info[idx].get("done"):
            raise ValueError(f"Environment {idx} has finished.")


server = TextWorldExpressServer()
```

Reward design note:

- TextWorldExpress `step()` returns a score-like reward value.
- For RL, use `score` as the reward at first.
- Later we can improve reward shaping, for example:
  - final success bonus
  - invalid action penalty
  - step penalty
  - normalized score delta instead of raw score

## Step 6: FastAPI Server

`agentenv_textworldexpress/server.py`:

```python
from fastapi import FastAPI
from .model import StepRequestBody, ResetRequestBody, CloseRequestBody
from .environment import server

app = FastAPI()


@app.get("/")
def hello():
    return "This is environment TextWorldExpress."


@app.post("/create")
def create():
    return server.create()


@app.post("/step")
def step(body: StepRequestBody):
    return server.step(body.id, body.action)


@app.post("/reset")
def reset(body: ResetRequestBody):
    return server.reset(body.id, body.data_idx)


@app.get("/observation")
def get_observation(id: int):
    return server.observe(id)


@app.post("/close")
def close(body: CloseRequestBody):
    return server.close(body.id)
```

`agentenv_textworldexpress/launch.py`:

```python
import argparse
import uvicorn


def launch():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=36002)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    uvicorn.run("agentenv_textworldexpress:app", host=args.host, port=args.port)
```

`agentenv_textworldexpress/__init__.py`:

```python
from .server import app
from .launch import launch
```

## Step 7: AgentEnv Client

Add a new file:

```text
AgentGym/agentenv/agentenv/envs/textworldexpress.py
```

Skeleton:

```python
from typing import Any, Mapping
import re
import requests
from requests.exceptions import RequestException

from agentenv.controller import BaseEnvClient, BaseTask
from agentenv.controller.types import ConversationMessage, StepOutput


class TextWorldExpressEnvClient(BaseEnvClient):
    conversation_start = (
        ConversationMessage({
            "from": "human",
            "loss": None,
            "value": (
                "You are playing a text adventure game. Every round you will receive "
                "an observation and a list of valid actions. Think briefly, then choose "
                "exactly one valid action.\n\n"
                "Your output must strictly follow this format:\n"
                "Thought:\n<your thoughts>\n\nAction:\n<one valid action>"
            ),
        }),
        ConversationMessage({
            "from": "gpt",
            "loss": False,
            "value": "OK. I will choose one valid action each round.",
        }),
    )

    def __init__(self, env_server_base: str, data_len: int, *args, timeout: int = 300, **kwargs):
        super().__init__(*args, **kwargs)
        self.env_server_base = env_server_base
        self.timeout = timeout
        self.data_len = data_len

        ok = requests.post(f"{self.env_server_base}/create", timeout=self.timeout)
        if ok.status_code != 200:
            raise RequestException(f"Failed to create environment: {ok}")
        self.env_id = ok.json()["id"]
        self.info = {"observation": "", "reward": 0, "done": False}

    def __len__(self):
        return self.data_len

    def _post(self, path: str, data: dict[str, Any]) -> dict[str, Any]:
        data["id"] = self.env_id
        res = requests.post(f"{self.env_server_base}/{path}", json=data, timeout=self.timeout)
        assert res.status_code == 200
        return res.json()

    def observe(self) -> str:
        return self.info["observation"]

    def step(self, action: str) -> StepOutput:
        action_matches = re.findall(r"Action:\s*(.*?)(?=\n|$)", action, re.DOTALL)
        if len(action_matches) > 1:
            return StepOutput(
                state="Error: Only one 'Action' is allowed per response.",
                reward=0,
                done=False,
            )
        action = action_matches[-1].strip() if action_matches else ""
        response = self._post("step", {"action": action})
        self.info = {
            "observation": response["observation"],
            "reward": response["reward"],
            "done": response["done"],
        }
        return StepOutput(
            state=response["observation"],
            reward=response["score"],
            done=response["done"],
        )

    def reset(self, data_idx: int = 0) -> dict[str, Any]:
        response = self._post("reset", {"data_idx": data_idx})
        self.info = {
            "observation": response["observation"],
            "reward": response["reward"],
            "done": response["done"],
        }
        return response

    def close(self):
        return self._post("close", {})


class TextWorldExpressTask(BaseTask):
    env_client_cls = TextWorldExpressEnvClient
    env_name = "TextWorldExpress"

    def __init__(self, client_args: Mapping[str, Any], *args, n_clients: int = 1, **kwargs):
        super().__init__(client_args, n_clients, *args, **kwargs)
```

Then update:

```text
AgentGym/agentenv/agentenv/envs/__init__.py
```

Add:

```python
from .textworldexpress import TextWorldExpressEnvClient, TextWorldExpressTask
```

Then update:

```text
AgentGym-RL/verl/utils/agentgym/client.py
```

Add the import:

```python
TextWorldExpressEnvClient,
```

Add the registry entry:

```python
"textworldexpress": TextWorldExpressEnvClient,
```

## Step 8: Install Editable Packages

After adding files:

```bash
conda activate agentgym-rl
python -m pip install -e AgentGym/agentenv
python -m pip install -e AgentGym/agentenv-textworldexpress
```

## Step 9: Smoke Test Server

Start server:

```bash
textworldexpress --host 127.0.0.1 --port 36002
```

In another terminal:

```bash
python - <<'PY'
import requests

base = "http://127.0.0.1:36002"
print(requests.get(base + "/").text)

env_id = requests.post(base + "/create").json()["id"]
print("id:", env_id)

obs = requests.post(base + "/reset", json={"id": env_id, "data_idx": 0}).json()
print(obs["observation"][:1000])

step = requests.post(base + "/step", json={"id": env_id, "action": "look around"}).json()
print(step)

print(requests.post(base + "/close", json={"id": env_id}).json())
PY
```

## Step 10: Item ID Files

Create:

```text
AgentItemId/textworldexpress_train.json
AgentItemId/textworldexpress_test.json
```

Example:

```json
[
  {"item_id": "textworldexpress_0"},
  {"item_id": "textworldexpress_1"},
  {"item_id": "textworldexpress_2"},
  {"item_id": "textworldexpress_3"},
  {"item_id": "textworldexpress_4"},
  {"item_id": "textworldexpress_5"},
  {"item_id": "textworldexpress_6"},
  {"item_id": "textworldexpress_7"}
]
```

## Step 11: Training Script

We can reuse the BabyAI training script pattern, changing only:

```bash
task_name="textworldexpress"
env_server_url="${ENV_SERVER_URL:-http://127.0.0.1:36002}"
train_file="${TRAIN_FILE:-AgentItemId/textworldexpress_train.json}"
```

Recommended first smoke settings:

```bash
TRAIN_BATCH_SIZE=4
ROLLOUT_N=2
ROUNDS=6
TOTAL_EPOCHS=1
```

TextWorldExpress observations and valid-action lists can be long, so keep the first run small.

## Step 12: Slurm Smoke Job

Use a new Slurm script like:

```text
slurm/textworldexpress_rl_smoke.sbatch
```

Differences from BabyAI:

```bash
export ENV_SERVER_URL="http://127.0.0.1:36002"
export EXP_NAME="textworldexpress_smoke_${SLURM_JOB_ID}"
export MODEL_SAVE_PATH="saves/${EXP_NAME}"
export TRAIN_FILE="AgentItemId/textworldexpress_train.json"

textworldexpress --host 127.0.0.1 --port 36002 > "${MODEL_SAVE_PATH}/textworldexpress_server.log" 2>&1 &
```

Everything else can follow the BabyAI Slurm script.

## Expected Issues

1. Java missing or wrong version

Check:

```bash
java -version
```

On Quest this worked:

```bash
module load 'java/jdk-17.0.2+8'
java -version
```

2. Py4J port collision

The installed version uses dynamic Py4J ports and does not support `threadNum`. If startup is slow, reduce RL smoke `TRAIN_BATCH_SIZE`; each rollout client creates a TextWorldExpress backend.

3. Invalid actions from the LLM

Include valid actions in the observation and ask the model to choose exactly one. Later, add invalid action penalty.

4. Reward too sparse or unstable

Start with raw `score`. Then try score deltas or terminal success bonus.

5. Item ID naming mismatch

Make sure every data item starts with:

```text
textworldexpress_
```

and `client.py` registers:

```python
"textworldexpress": TextWorldExpressEnvClient
```

## Implementation Order I Recommend

1. Install and run the pure TextWorldExpress Python smoke test.
2. Implement the FastAPI server package.
3. Test `/create`, `/reset`, `/step`, `/close` manually.
4. Add AgentGym client registration.
5. Run an AgentGym client smoke test.
6. Run one Slurm RL smoke job.
7. Only then increase training size.

This gives a clean bonus story:

```text
We added a new text-game environment to AgentGym-RL, verified standalone interaction, wrapped it in the AgentEnv protocol, and ran the same LLM RL training framework on it.
```

## What Has Been Implemented Here

Files added:

```text
AgentGym/agentenv-textworldexpress/
AgentGym/agentenv/agentenv/envs/textworldexpress.py
AgentItemId/textworldexpress_train.json
AgentItemId/textworldexpress_test.json
examples/train/AgentGym-RL/textworldexpress_smoke_local.sh
slurm/textworldexpress_rl_smoke.sbatch
```

Files updated:

```text
AgentGym/agentenv/agentenv/envs/__init__.py
AgentGym-RL/verl/utils/agentgym/client.py
```

Dependencies installed into `agentgym-rl`:

```bash
python -m pip install textworld-express py4j
python -m pip install -e AgentGym/agentenv -e AgentGym/agentenv-textworldexpress
```

Smoke tests already passed:

```bash
module load 'java/jdk-17.0.2+8'
conda run -n agentgym-rl python -c "from textworld_express import TextWorldExpressEnv; print(TextWorldExpressEnv)"
```

The pure TextWorldExpress API successfully ran `reset()` and `step()`.

The AgentEnv HTTP server successfully ran:

```text
GET /                    -> "This is environment TextWorldExpress."
POST /create             -> {"id": ...}
POST /reset              -> observation + valid_actions
POST /step               -> observation + score + done
POST /close              -> true
```

The AgentGym-RL client registry also worked with:

```text
task_name="textworldexpress"
```

## Run Log: `7103978`

TextWorldExpress RL smoke job:

```text
Job id: 7103978
Experiment: textworldexpress_smoke_7103978
Output: saves/textworldexpress_smoke_7103978/
W&B offline run: AgentGym-RL/wandb/offline-run-20260503_185117-dlen2nvw
```

This run completed the training script successfully. The Slurm log reached W&B summary and killed the env server cleanly at the end.

Training metrics from `slurm_logs/twx_smoke-7103978.out`:

```text
step 1: actor/kl_loss=0.002, actor/entropy_loss=1.775, critic/score/mean=0.016, critic/rewards/mean=0.016, timing_s/step=34.633
step 2: actor/kl_loss=0.001, actor/entropy_loss=1.750, critic/score/mean=0.000, critic/rewards/mean=0.000, timing_s/step=29.768
step 3: actor/kl_loss=0.001, actor/entropy_loss=1.608, critic/score/mean=0.000, critic/rewards/mean=0.000, timing_s/step=22.676
```

Rollout files:

```text
saves/textworldexpress_smoke_7103978/executer_logs/step1/0.json
saves/textworldexpress_smoke_7103978/executer_logs/step2/0.json
saves/textworldexpress_smoke_7103978/executer_logs/step3/0.json
```

Case-study finding:

The base model often sees the valid-action list, but still invents actions that are not legal, for example:

```text
Action: open cabinet
Action: nothing decision necessary
Action: put clean blue bra in wardrobe
```

The environment then returns:

```text
Unknown action: I'm not sure what you mean.
```

This is a useful pilot finding: TextWorldExpress exposes a clear action-grounding problem for LLM agents.

## Baseline Improvement: Action Canonicalization

After `7103978`, the client was improved in:

```text
AgentGym/agentenv/agentenv/envs/textworldexpress.py
```

Changes:

- Prompt now says the action must be copied verbatim from the valid-action list.
- The client caches `valid_actions`.
- If the model gives an action with different capitalization, it maps it back to the canonical valid action.
- If the model omits `Action:` but mentions exactly one valid action in its free-text response, the client extracts that valid action.

Verification:

```text
Input:  I think I should take clean blue bra now.
Parsed action: take clean blue bra
Result: You take the clean blue bra.
Reward: 0.125
```

This is a small baseline improvement before running a second smoke job.

## Run Log: `7107157`

Second TextWorldExpress smoke job after action canonicalization:

```text
Job id: 7107157
Experiment: textworldexpress_smoke_7107157
Output: saves/textworldexpress_smoke_7107157/
W&B offline run: AgentGym-RL/wandb/offline-run-20260503_200239-qrknfqx9
```

This run also completed successfully. `squeue -j 7107157` no longer shows a row, and the log reached W&B summary.

Training metrics:

```text
step 1: actor/kl_loss=0.001, actor/entropy_loss=1.896, critic/score/mean=0.000, critic/rewards/mean=0.000, timing_s/step=20.765
step 2: actor/kl_loss=0.001, actor/entropy_loss=1.503, critic/score/mean=0.016, critic/rewards/mean=0.016, timing_s/step=20.545
step 3: actor/kl_loss=0.002, actor/entropy_loss=1.660, critic/score/mean=0.000, critic/rewards/mean=0.000, timing_s/step=19.489
```

Rollout files:

```text
saves/textworldexpress_smoke_7107157/executer_logs/step1/0.json
saves/textworldexpress_smoke_7107157/executer_logs/step2/0.json
saves/textworldexpress_smoke_7107157/executer_logs/step3/0.json
```

Quick comparison:

```text
7103978 before canonicalization: Unknown action count = 54
7107157 after canonicalization:  Unknown action count = 47
```

This is a modest improvement, not a solved problem. The remaining failures suggest that the model often:

- selects plausible but nonexistent containers or objects,
- ignores the exact valid-action string,
- produces assistant text without a usable `Action:` line.

Next recommended improvement:

1. Add an invalid-action penalty in the client or server response.
2. Try constrained action selection: choose among valid actions with a smaller prompt or postprocessor.
3. Increase run length only after the invalid-action rate is lower.

## Final Artifact Status

Generated assignment artifacts are in:

```bash
pilot_outputs/final_artifacts/
```

Relevant files:

```bash
pilot_outputs/final_artifacts/textworldexpress_metrics.csv
pilot_outputs/final_artifacts/textworldexpress_training_curves.png
pilot_outputs/final_artifacts/smoke_comparison_training_curves.png
pilot_outputs/final_artifacts/case_studies.md
pilot_outputs/final_artifacts/completion_report.md
```

Regenerate with:

```bash
conda run -n agentgym-rl python scripts/build_pilot_artifacts.py
```

## Run the RL Smoke Job

Submit:

```bash
sbatch slurm/textworldexpress_rl_smoke.sbatch
```

Check:

```bash
squeue -j <jobid>
sacct -j <jobid> --format=JobID,JobName,State,ExitCode,Elapsed,NodeList
tail -n 120 slurm_logs/twx_smoke-<jobid>.err
tail -n 120 slurm_logs/twx_smoke-<jobid>.out
```

Expected output directory:

```text
saves/textworldexpress_smoke_<jobid>/
```
