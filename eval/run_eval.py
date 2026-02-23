"""Evaluation script sweeping agent counts and LiDAR settings with CSV output."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from shepherd_env import EnvConfig, ShepherdEnv


def evaluate_episode(env: ShepherdEnv, horizon: int = 400) -> dict[str, float]:
    """Roll one episode with baseline-like heuristic actions."""

    obs = env.reset()
    total_energy = 0.0
    messages = 0
    for t in range(horizon):
        act = {f"dog_{j}": env._baseline_dog_action(j) for j in range(env.config.n_dogs)}
        obs, _, done, info = env.step(act)
        total_energy += info["energy"]
        messages += info["messages"]
        if done:
            break
    return {
        "success": float(info["enclosure_fraction"] >= 0.95),
        "herd_time": float(t + 1),
        "collisions": float(info["collisions"]),
        "messages": float(messages),
        "energy": float(total_energy),
    }


def main() -> None:
    """Run grid evaluation and write episode-level metrics to CSV."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--out", type=Path, default=Path("eval_metrics.csv"))
    args = parser.parse_args()

    rows = []
    for n_a in [1, 5, 10]:
        for n_d in [1, 3, 5]:
            for fov in [60.0, 90.0, 120.0]:
                for n_rays in [36, 90, 180]:
                    cfg = EnvConfig(n_sheep=n_a, n_dogs=n_d, sheep_fov_deg=fov, sheep_n_rays=n_rays)
                    env = ShepherdEnv(cfg)
                    for _ in range(args.episodes):
                        m = evaluate_episode(env)
                        m.update({"n_sheep": n_a, "n_dogs": n_d, "fov": fov, "n_rays": n_rays})
                        rows.append(m)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
