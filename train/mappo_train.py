"""Starter multi-agent policy-gradient loop that can learn without demonstration data."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from shepherd_env import EnvConfig, ShepherdEnv


@dataclass
class GaussianActor:
    """Simple Gaussian policy with linear mean and fixed std."""

    w: np.ndarray
    b: np.ndarray
    log_std: float = -0.2

    @classmethod
    def init(cls, in_dim: int, out_dim: int = 2, seed: int = 0) -> "GaussianActor":
        """Create random actor parameters."""

        rng = np.random.default_rng(seed)
        return cls(w=rng.normal(0.0, 0.05, size=(out_dim, in_dim)), b=np.zeros(out_dim))

    def mean(self, x: np.ndarray) -> np.ndarray:
        """Return deterministic action mean for one observation vector."""

        return self.w @ x + self.b

    def sample(self, x: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Sample action and noise for score-function gradients."""

        mu = self.mean(x)
        std = np.exp(self.log_std)
        eps = rng.normal(0.0, 1.0, size=mu.shape)
        return mu + std * eps, eps

    def deterministic(self, x: np.ndarray) -> np.ndarray:
        """Return deterministic policy action."""

        return self.mean(x)


class RunningNorm:
    """Online feature normalizer for stable learning."""

    def __init__(self, dim: int) -> None:
        self.count = 1e-6
        self.mean = np.zeros(dim)
        self.var = np.ones(dim)

    def update(self, x: np.ndarray) -> None:
        """Update running mean/variance with batch vectors."""

        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + (delta * delta) * self.count * batch_count / total
        self.mean = new_mean
        self.var = m2 / total
        self.count = total

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize using running stats."""

        return (x - self.mean) / np.sqrt(self.var + 1e-6)


def flatten_dog_obs(obs: dict) -> np.ndarray:
    """Flatten structured dog observation into vector."""

    return np.concatenate([obs["pos"], obs["vel"], obs["relative_sheep"].reshape(-1), obs["llm_intent"]], axis=0)


def discounted_returns(rewards: list[float], gamma: float) -> np.ndarray:
    """Compute discounted returns for one trajectory."""

    out = np.zeros(len(rewards), dtype=float)
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        out[t] = running
    return out


def run_training(
    iters: int,
    steps_per_iter: int,
    seed: int,
    gamma: float,
    lr: float,
    exploration: float,
    use_curriculum: bool,
) -> dict:
    """Run a lightweight REINFORCE-style multi-agent loop from scratch."""

    rng = np.random.default_rng(seed)
    cfg = EnvConfig(seed=seed)
    env = ShepherdEnv(cfg)
    obs0 = env.reset(seed=seed)
    in_dim = flatten_dog_obs(obs0["dog_0"]).shape[0]
    actor = GaussianActor.init(in_dim=in_dim, seed=seed)
    actor.log_std = float(np.log(max(1e-3, exploration)))
    norm = RunningNorm(in_dim)

    history = {
        "iter": [],
        "episode_reward": [],
        "success_rate": [],
        "mean_enclosure": [],
        "n_sheep": [],
        "n_dogs": [],
    }

    for it in range(iters):
        if use_curriculum:
            n_sheep = 1 if it < iters // 3 else (3 if it < 2 * iters // 3 else 5)
            n_dogs = 3 if it < 2 * iters // 3 else 5
            fov = 120.0 if it < iters // 2 else 90.0
            env_cfg = {"n_sheep": n_sheep, "n_dogs": n_dogs, "sheep_fov_deg": fov}
        else:
            n_sheep = cfg.n_sheep
            n_dogs = cfg.n_dogs
            env_cfg = None

        obs = env.reset(seed=seed + it, config=env_cfg)
        traj_x: list[np.ndarray] = []
        traj_eps: list[np.ndarray] = []
        traj_r: list[float] = []
        total_reward = 0.0
        successes = 0
        enclosures: list[float] = []

        for _ in range(steps_per_iter):
            xs = np.array([flatten_dog_obs(obs[f"dog_{j}"]) for j in range(env.config.n_dogs)], dtype=float)
            norm.update(xs)
            xs_n = np.array([norm.normalize(x) for x in xs])
            act_dict: dict[str, np.ndarray] = {}
            eps_bank: list[np.ndarray] = []
            for j in range(env.config.n_dogs):
                a, eps = actor.sample(xs_n[j], rng)
                a = np.clip(a, -1.0, 1.0) * env.config.u_max_dog
                act_dict[f"dog_{j}"] = a.astype(float)
                eps_bank.append(eps)
            next_obs, rewards, done, info = env.step(act_dict)
            team_r = float(np.mean([rewards[f"dog_{j}"] for j in range(env.config.n_dogs)]))
            total_reward += team_r
            enclosures.append(float(info.get("enclosure_fraction", 0.0)))
            if done and info.get("enclosure_fraction", 0.0) >= 0.95:
                successes += 1
            for j in range(env.config.n_dogs):
                traj_x.append(xs_n[j])
                traj_eps.append(eps_bank[j])
                traj_r.append(team_r)
            obs = next_obs
            if done:
                break

        returns = discounted_returns(traj_r, gamma)
        adv = (returns - returns.mean()) / (returns.std() + 1e-6)
        std = np.exp(actor.log_std)
        grad_w = np.zeros_like(actor.w)
        grad_b = np.zeros_like(actor.b)
        for x, eps, a in zip(traj_x, traj_eps, adv):
            grad_mu = (eps / std) * a
            grad_w += np.outer(grad_mu, x)
            grad_b += grad_mu
        scale = 1.0 / max(1, len(traj_x))
        actor.w += lr * grad_w * scale
        actor.b += lr * grad_b * scale

        history["iter"].append(it)
        history["episode_reward"].append(total_reward)
        history["success_rate"].append(float(successes > 0))
        history["mean_enclosure"].append(float(np.mean(enclosures) if enclosures else 0.0))
        history["n_sheep"].append(n_sheep)
        history["n_dogs"].append(n_dogs)

    return history


def main() -> None:
    """CLI for running lightweight scratch RL training without demonstrations."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--exploration", type=float, default=0.35)
    parser.add_argument("--curriculum", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("train_history.json"))
    args = parser.parse_args()
    history = run_training(
        iters=args.iters,
        steps_per_iter=args.steps,
        seed=args.seed,
        gamma=args.gamma,
        lr=args.lr,
        exploration=args.exploration,
        use_curriculum=args.curriculum,
    )
    args.out.write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
