#!/usr/bin/env python3
"""CPU-only BabyAI pilot artifact generator.

This is not a replacement for the full AgentGym-RL PPO/GRPO training run. It is
a lightweight harness for presentation prep when a GPU/vLLM stack is not
available: it verifies environment interaction, collects trajectories, and
plots reward curves for simple policies.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import requests

from agentenv.envs.babyai import BabyAIEnvClient


ACTION_RE = re.compile(r"Available actions:\s*\[(.*?)\]", re.DOTALL)
GOAL_RE = re.compile(r"Your goal:\s*(.*?)(?:\n|$)", re.DOTALL)


@dataclass
class StepRecord:
    turn: int
    observation: str
    thought: str
    action: str
    reward: float
    done: bool


def parse_actions(observation: str) -> list[str]:
    match = ACTION_RE.search(observation)
    if not match:
        return []
    return re.findall(r'"([^"]+)"', match.group(1))


def parse_goal(observation: str) -> str:
    match = GOAL_RE.search(observation)
    return " ".join(match.group(1).split()) if match else ""


def choose_random(actions: list[str], rng: random.Random) -> tuple[str, str]:
    candidates = [a for a in actions if a != "check available actions"] or actions
    action = rng.choice(candidates) if candidates else "check available actions"
    thought = "I will explore using one valid action from the current action list."
    return thought, action


def choose_heuristic(
    goal: str, actions: list[str], rng: random.Random, exploration_rate: float
) -> tuple[str, str]:
    if rng.random() < exploration_rate:
        thought, action = choose_random(actions, rng)
        return thought + " This is an exploration step.", action

    goal_lower = goal.lower()
    prioritized: list[str] = []

    # BabyAI macro actions often expose exactly the useful high-level action,
    # such as "go to red ball 1" or "pickup grey key 1".
    if goal_lower.startswith("go to "):
        target = goal_lower.removeprefix("go to ").removeprefix("the ").strip()
        prioritized.extend(
            a for a in actions if a.startswith("go to ") and target in a
        )
    if goal_lower.startswith("pick up ") or goal_lower.startswith("pickup "):
        target = (
            goal_lower.replace("pick up ", "")
            .replace("pickup ", "")
            .removeprefix("the ")
            .strip()
        )
        prioritized.extend(
            a for a in actions if a.startswith("pickup ") and target in a
        )
    if goal_lower.startswith("open "):
        target = goal_lower.removeprefix("open ").removeprefix("the ").strip()
        prioritized.extend(
            a
            for a in actions
            if ("toggle" in a or "go through" in a) and target in a
        )

    for action in prioritized:
        thought = (
            f"The goal says '{goal}', and the valid actions include a matching "
            f"macro action: '{action}'. I should use that direct action."
        )
        return thought, action

    if "check available actions" in actions:
        return (
            "I cannot map the goal to a direct macro action yet, so I should inspect the valid actions.",
            "check available actions",
        )
    return choose_random(actions, rng)


def format_agent_message(thought: str, action: str) -> str:
    return f"Thought:\n{thought}\n\nAction:\n{action}"


def run_episode(
    base_url: str,
    data_idx: int,
    policy: str,
    max_rounds: int,
    rng: random.Random,
    exploration_rate: float,
) -> dict:
    client = BabyAIEnvClient(base_url, data_len=1, timeout=60)
    records: list[StepRecord] = []
    try:
        reset = client.reset(data_idx)
        observation = reset["observation"]
        goal = parse_goal(observation)
        total_reward = float(reset.get("score", 0))
        done = bool(reset.get("done", False))

        for turn in range(max_rounds):
            actions = parse_actions(observation)
            if policy == "random":
                thought, action = choose_random(actions, rng)
            elif policy == "heuristic":
                thought, action = choose_heuristic(goal, actions, rng, exploration_rate)
            else:
                raise ValueError(f"Unknown policy: {policy}")

            output = client.step(format_agent_message(thought, action))
            total_reward = float(output.reward)
            done = bool(output.done)
            records.append(
                StepRecord(
                    turn=turn,
                    observation=observation,
                    thought=thought,
                    action=action,
                    reward=total_reward,
                    done=done,
                )
            )
            observation = output.state
            if done:
                break

        return {
            "data_idx": data_idx,
            "policy": policy,
            "goal": goal,
            "score": total_reward,
            "success": done,
            "turns": len(records),
            "trajectory": [r.__dict__ for r in records],
        }
    finally:
        client.close()


def check_server(base_url: str) -> None:
    response = requests.get(base_url, timeout=5)
    response.raise_for_status()


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def write_metrics(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["episode", "data_idx", "policy", "goal", "score", "success", "turns"]
        )
        writer.writeheader()
        for episode, row in enumerate(rows):
            writer.writerow(
                {
                    "episode": episode,
                    "data_idx": row["data_idx"],
                    "policy": row["policy"],
                    "goal": row["goal"],
                    "score": row["score"],
                    "success": row["success"],
                    "turns": row["turns"],
                }
            )


def plot_curves(path: Path, rows: list[dict]) -> None:
    by_policy: dict[str, list[dict]] = {}
    for row in rows:
        by_policy.setdefault(row["policy"], []).append(row)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=160)
    for policy, policy_rows in by_policy.items():
        episodes = list(range(len(policy_rows)))
        rewards = [float(r["score"]) for r in policy_rows]
        successes = [1.0 if r["success"] else 0.0 for r in policy_rows]
        axes[0].plot(episodes, rewards, marker="o", linewidth=1.5, label=policy)
        axes[1].plot(episodes, successes, marker="o", linewidth=1.5, label=policy)

    axes[0].set_title("BabyAI Episode Reward")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Score")
    axes[0].grid(alpha=0.3)
    axes[1].set_title("BabyAI Success")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Success")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(alpha=0.3)
    for axis in axes:
        axis.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def write_case_study(path: Path, rows: list[dict]) -> None:
    success = next((r for r in rows if r["policy"] == "heuristic" and r["success"]), None)
    failure = next((r for r in rows if r["policy"] == "random" and not r["success"]), None)
    examples = [("Successful heuristic case", success), ("Random-policy failure case", failure)]

    lines = ["# BabyAI Pilot Case Study", ""]
    for title, row in examples:
        if row is None:
            continue
        lines.extend(
            [
                f"## {title}",
                "",
                f"- data_idx: `{row['data_idx']}`",
                f"- goal: `{row['goal']}`",
                f"- final score: `{row['score']}`",
                f"- success: `{row['success']}`",
                f"- turns: `{row['turns']}`",
                "",
            ]
        )
        for step in row["trajectory"][:5]:
            obs = " ".join(step["observation"].split())
            if len(obs) > 450:
                obs = obs[:450] + "..."
            lines.extend(
                [
                    f"### Turn {step['turn']}",
                    "",
                    f"Observation: {obs}",
                    "",
                    f"Thought: {step['thought']}",
                    "",
                    f"Action: `{step['action']}`",
                    "",
                    f"Reward after action: `{step['reward']}`, done: `{step['done']}`",
                    "",
                ]
            )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:36001")
    parser.add_argument("--episodes", type=int, default=12)
    parser.add_argument("--max-rounds", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", default="pilot_outputs/babyai")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    check_server(args.base_url)

    rows: list[dict] = []
    for policy in ["random", "heuristic"]:
        for episode in range(args.episodes):
            # Use the easier first BabyAI level for a clear pilot case study, while
            # varying seeds across episodes.
            data_idx = episode * 40
            exploration_rate = max(0.0, 0.25 * (1 - episode / max(1, args.episodes - 1)))
            rows.append(
                run_episode(
                    args.base_url,
                    data_idx=data_idx,
                    policy=policy,
                    max_rounds=args.max_rounds,
                    rng=rng,
                    exploration_rate=exploration_rate if policy == "heuristic" else 1.0,
                )
            )

    write_jsonl(out_dir / "trajectories.jsonl", rows)
    write_metrics(out_dir / "metrics.csv", rows)
    plot_curves(out_dir / "reward_curve.png", rows)
    write_case_study(out_dir / "case_study.md", rows)

    summary = {}
    for policy in ["random", "heuristic"]:
        policy_rows = [r for r in rows if r["policy"] == policy]
        summary[policy] = {
            "episodes": len(policy_rows),
            "avg_score": sum(float(r["score"]) for r in policy_rows) / len(policy_rows),
            "success_rate": sum(1 for r in policy_rows if r["success"]) / len(policy_rows),
            "avg_turns": sum(int(r["turns"]) for r in policy_rows) / len(policy_rows),
        }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote artifacts to {out_dir}")


if __name__ == "__main__":
    main()
