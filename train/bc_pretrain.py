"""Optional behavior cloning pretraining with synthetic fallback data generation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from shepherd_env import EnvConfig, ShepherdEnv
from train.mappo_train import GaussianActor, flatten_dog_obs


def load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load npz trajectory dataset containing obs and acts arrays."""

    data = np.load(path)
    return data["obs"], data["acts"]


def collect_synthetic_dataset(seed: int, samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Collect pseudo-demonstrations from the built-in baseline controller."""

    env = ShepherdEnv(EnvConfig(seed=seed))
    obs = env.reset(seed=seed)
    obs_bank: list[np.ndarray] = []
    act_bank: list[np.ndarray] = []
    while len(obs_bank) < samples:
        actions = {}
        for j in range(env.config.n_dogs):
            x = flatten_dog_obs(obs[f"dog_{j}"])
            a = env._baseline_dog_action(j) / env.config.u_max_dog
            obs_bank.append(x)
            act_bank.append(np.clip(a, -1.0, 1.0))
            actions[f"dog_{j}"] = env._baseline_dog_action(j)
            if len(obs_bank) >= samples:
                break
        obs, _, done, _ = env.step(actions)
        if done:
            obs = env.reset(seed=seed + len(obs_bank))
    return np.array(obs_bank), np.array(act_bank)


def train_bc(obs: np.ndarray, acts: np.ndarray, epochs: int = 40, lr: float = 1e-2, seed: int = 0) -> GaussianActor:
    """Train a linear Gaussian actor with supervised gradients."""

    actor = GaussianActor.init(in_dim=obs.shape[1], seed=seed)
    for _ in range(epochs):
        pred = obs @ actor.w.T + actor.b
        err = pred - acts
        grad_w = (err.T @ obs) * (2.0 / len(obs))
        grad_b = err.mean(axis=0) * 2.0
        actor.w -= lr * grad_w
        actor.b -= lr * grad_b
    return actor


def main() -> None:
    """CLI for optional BC pretraining without requiring external demos."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=None)
    parser.add_argument("--synthetic-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--out", type=Path, default=Path("bc_actor.npz"))
    args = parser.parse_args()

    if args.data is not None and args.data.exists():
        obs, acts = load_dataset(args.data)
    else:
        obs, acts = collect_synthetic_dataset(seed=args.seed, samples=args.synthetic_samples)

    actor = train_bc(obs, acts, seed=args.seed)
    np.savez(args.out, w=actor.w, b=actor.b, log_std=actor.log_std)


if __name__ == "__main__":
    main()
