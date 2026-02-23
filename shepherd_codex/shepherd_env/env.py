"""Gym-like shepherding environment for small flock StringNet experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import yaml

from .dynamics import semi_implicit_euler_step
from .spawn import spawn_dogs, spawn_sheep


@dataclass
class EnvState:
    sheep_pos: np.ndarray
    sheep_vel: np.ndarray
    dog_pos: np.ndarray
    dog_vel: np.ndarray


class ShepherdEnv:
    """Minimal gym-like environment with reset/step and deterministic state."""

    def __init__(self, config_path: str = "shepherd_codex/configs/default.yaml") -> None:
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        self.rng = np.random.default_rng(self.config.get("seed", 0))
        self.goal_region = np.array([[12.0, 8.0], [20.0, 12.0]])
        self.state = None
        self.t = 0

    def reset(self, seed: int | None = None, config: dict | None = None) -> dict[str, Any]:
        if config:
            self.config.update(config)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        na = int(self.config.get("N_a", 5))
        nd = int(self.config.get("N_d", 3))
        sheep_pos, center, radius = spawn_sheep(na, self.rng, tuple(self.config.get("rho_ac_range", [0.6, 1.2])), self.config.get("d_min", 0.15))
        dog_pos = spawn_dogs(nd, self.rng)
        self.state = EnvState(
            sheep_pos=sheep_pos,
            sheep_vel=np.zeros((na, 2), dtype=float),
            dog_pos=dog_pos,
            dog_vel=np.zeros((nd, 2), dtype=float),
        )
        self.t = 0
        return self.get_state() | {"r_ac": center, "rho_ac": radius}

    def get_state(self) -> dict[str, Any]:
        assert self.state is not None
        goal_center = np.array([16.0, 10.0])
        return {
            "sheep_pos": self.state.sheep_pos.copy(),
            "sheep_vel": self.state.sheep_vel.copy(),
            "dog_pos": self.state.dog_pos.copy(),
            "dog_vel": self.state.dog_vel.copy(),
            "goal": goal_center,
            "goal_region": self.goal_region.copy(),
            "dt": self.config["dt"],
            "r_agent": self.config["r_agent"],
        }

    def step(self, action_dict: dict[int, np.ndarray]) -> tuple[dict, dict, bool, dict]:
        assert self.state is not None
        nd = self.state.dog_pos.shape[0]
        na = self.state.sheep_pos.shape[0]
        dog_u = np.zeros((nd, 2), dtype=float)
        for j in range(nd):
            dog_u[j] = np.asarray(action_dict.get(j, np.zeros(2)), dtype=float)
        self.state.dog_pos, self.state.dog_vel = semi_implicit_euler_step(
            self.state.dog_pos,
            self.state.dog_vel,
            dog_u,
            self.config["C_D"],
            self.config["ubar_d"],
            self.config["dt"],
        )
        # sheep drift towards goal and away from nearby dogs
        goal = np.array([16.0, 10.0])
        sheep_u = np.zeros((na, 2), dtype=float)
        for i in range(na):
            to_goal = goal - self.state.sheep_pos[i]
            if np.linalg.norm(to_goal) > 1e-8:
                sheep_u[i] += 0.6 * to_goal / np.linalg.norm(to_goal)
            for d in self.state.dog_pos:
                diff = self.state.sheep_pos[i] - d
                dist = np.linalg.norm(diff)
                if dist < 2.0 and dist > 1e-6:
                    sheep_u[i] += 0.8 * diff / (dist**2)
        self.state.sheep_pos, self.state.sheep_vel = semi_implicit_euler_step(
            self.state.sheep_pos,
            self.state.sheep_vel,
            sheep_u,
            self.config["C_D"],
            self.config["ubar_a"],
            self.config["dt"],
        )
        self.state.sheep_pos = np.clip(self.state.sheep_pos, [0, 0], [20, 20])
        self.state.dog_pos = np.clip(self.state.dog_pos, [0, 0], [20, 20])

        self.t += 1
        obs = self.get_state()
        inside = (
            (obs["sheep_pos"][:, 0] >= self.goal_region[0, 0])
            & (obs["sheep_pos"][:, 0] <= self.goal_region[1, 0])
            & (obs["sheep_pos"][:, 1] >= self.goal_region[0, 1])
            & (obs["sheep_pos"][:, 1] <= self.goal_region[1, 1])
        )
        done = bool(np.all(inside) or self.t >= int(self.config.get("T_max", 1000)))
        rewards = {"team": float(np.mean(inside))}
        info = {"success": bool(np.all(inside)), "t": self.t}
        return obs, rewards, done, info
