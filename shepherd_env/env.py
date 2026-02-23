"""Gym-like shepherding environment using LiDAR sheep perception and LLM-guided dogs."""

from __future__ import annotations

import copy
import logging
import math
from dataclasses import asdict
from typing import Any

import numpy as np

from .config import EnvConfig
from .dynamics import chipade_panagou_enclosing_accel, reynolds_sheep_control, semi_implicit_euler
from .geometry import centroid, norm, point_in_polygon
from .lidar import LidarSensor
from .llm import MockLLMPlanner, intent_to_embedding, scene_tokens
from .shield import SafetyShield
from .spawn import sample_dogs, sample_points_in_disc

logger = logging.getLogger(__name__)


class ShepherdEnv:
    """Deterministic and seedable multi-agent shepherding environment."""

    def __init__(self, config: EnvConfig | None = None) -> None:
        self.config = config or EnvConfig()
        self.config.validate()
        self.rng = np.random.default_rng(self.config.seed)
        self.spawn_cfg: dict[str, Any] = {}
        self.lidar = LidarSensor(
            self.config.sheep_fov_deg,
            self.config.sheep_n_rays,
            self.config.sheep_lidar_range,
            self.config.sheep_lidar_sigma,
            self.config.sheep_false_negative,
            self.config.sheep_false_positive,
            self.config.sheep_intensity_noise,
            self.config.r_agent,
        )
        self.llm = MockLLMPlanner()
        self.shield = SafetyShield(self.config.u_max_dog, self.config.r_agent)
        self.step_count = 0
        self.sheep_pos = np.zeros((self.config.n_sheep, 2), dtype=float)
        self.sheep_vel = np.zeros((self.config.n_sheep, 2), dtype=float)
        self.sheep_heading = np.zeros(self.config.n_sheep, dtype=float)
        self.dog_pos = np.zeros((self.config.n_dogs, 2), dtype=float)
        self.dog_vel = np.zeros((self.config.n_dogs, 2), dtype=float)
        self.last_intent = "HOLD"

    def set_spawn_config(self, spawn_cfg: dict[str, Any]) -> None:
        """Set external spawn overrides for reset sampling."""

        self.spawn_cfg = copy.deepcopy(spawn_cfg)

    def _apply_config_override(self, override: dict[str, Any] | None) -> None:
        if override is None:
            return
        self.config = EnvConfig.from_overrides(self.config, override)
        self.lidar = LidarSensor(
            self.config.sheep_fov_deg,
            self.config.sheep_n_rays,
            self.config.sheep_lidar_range,
            self.config.sheep_lidar_sigma,
            self.config.sheep_false_negative,
            self.config.sheep_false_positive,
            self.config.sheep_intensity_noise,
            self.config.r_agent,
        )
        self.shield = SafetyShield(self.config.u_max_dog, self.config.r_agent)

    def reset(self, seed: int | None = None, config: dict[str, Any] | None = None) -> dict[str, dict[str, np.ndarray]]:
        """Reset environment and return initial observation dictionary."""

        self._apply_config_override(config)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        elif self.config.seed is not None:
            self.rng = np.random.default_rng(self.config.seed)

        center = np.array(self.spawn_cfg.get("spawn_center", self.config.spawn_center), dtype=float)
        radius = float(self.spawn_cfg.get("spawn_radius", self.config.spawn_radius))
        self.sheep_pos = sample_points_in_disc(self.rng, center, radius, self.config.n_sheep, self.config.min_pair_distance)
        self.sheep_vel = self.rng.normal(0.0, 0.1, size=(self.config.n_sheep, 2))
        self.sheep_heading = np.zeros(self.config.n_sheep, dtype=float)
        self.dog_pos = sample_dogs(
            self.rng,
            self.config.n_dogs,
            self.config.map_size,
            self.config.dog_spawn_mode,
            self.config.dog_spawn_nodes,
            center,
            radius,
        )
        self.dog_vel = np.zeros((self.config.n_dogs, 2), dtype=float)
        self.step_count = 0
        self.last_intent = "HOLD"
        logger.info("Environment reset with %d sheep and %d dogs", self.config.n_sheep, self.config.n_dogs)
        return self._get_obs()

    def _safe_polygon(self) -> list[tuple[float, float]]:
        g = np.array(self.config.goal, dtype=float)
        r = self.config.goal_radius
        return [(g[0] - r, g[1] - r), (g[0] + r, g[1] - r), (g[0] + r, g[1] + r), (g[0] - r, g[1] + r)]

    def _baseline_dog_action(self, dog_idx: int) -> np.ndarray:
        tokens = scene_tokens(self.sheep_pos, self.sheep_vel, self.config.goal)
        center = np.array(tokens["sheep_center"])
        goal = np.array(self.config.goal)
        axis = center - goal
        axis_n = axis / (norm(axis) + 1e-6)
        formation_r = max(1.0, tokens["sheep_spread"] + 1.0)
        phi = math.atan2(axis_n[1], axis_n[0])
        slot = phi + (2.0 * math.pi * dog_idx / max(1, self.config.n_dogs))
        desired = center + formation_r * np.array([math.cos(slot), math.sin(slot)])
        desired_vel = np.zeros(2)
        return chipade_panagou_enclosing_accel(self.dog_pos[dog_idx], self.dog_vel[dog_idx], desired, desired_vel, drag=self.config.drag)

    def _collisions(self) -> int:
        collisions = 0
        all_pos = np.concatenate([self.sheep_pos, self.dog_pos], axis=0)
        for i in range(len(all_pos)):
            d = np.linalg.norm(all_pos[i + 1 :] - all_pos[i], axis=1)
            collisions += int(np.sum(d < 2.0 * self.config.r_agent))
        return collisions

    def _sheep_in_goal(self) -> float:
        poly = self._safe_polygon()
        inside = sum(1 for p in self.sheep_pos if point_in_polygon(p, poly))
        return inside / max(1, self.config.n_sheep)

    def _goal_progress(self, prev_center: np.ndarray, new_center: np.ndarray) -> float:
        goal = np.array(self.config.goal, dtype=float)
        return float(np.linalg.norm(prev_center - goal) - np.linalg.norm(new_center - goal))

    def _get_obs(self) -> dict[str, dict[str, np.ndarray]]:
        obs: dict[str, dict[str, np.ndarray]] = {}
        tokens = scene_tokens(self.sheep_pos, self.sheep_vel, self.config.goal)
        intent, _ = self.llm.plan(tokens)
        self.last_intent = intent
        intent_vec = intent_to_embedding(intent)

        for i in range(self.config.n_sheep):
            scan = self.lidar.scan(
                self.sheep_pos[i],
                self.sheep_heading[i],
                self.sheep_pos,
                self.dog_pos,
                self.config.obstacles,
                i,
                self.rng,
            )
            item = {"lidar": scan, "vel": self.sheep_vel[i].copy()}
            if self.config.include_sheep_pos:
                item["pos"] = self.sheep_pos[i].copy()
            obs[f"sheep_{i}"] = item

        for j in range(self.config.n_dogs):
            rel = self.sheep_pos - self.dog_pos[j]
            d = np.linalg.norm(rel, axis=1)
            idx = np.argsort(d)[: self.config.k_nearest_sheep]
            pad = np.zeros((self.config.k_nearest_sheep, 2), dtype=float)
            used = rel[idx]
            pad[: len(used)] = used
            obs[f"dog_{j}"] = {
                "pos": self.dog_pos[j].copy(),
                "vel": self.dog_vel[j].copy(),
                "relative_sheep": pad,
                "llm_intent": intent_vec.copy(),
            }
        return obs

    def step(self, action_dict: dict[str, np.ndarray]) -> tuple[dict, dict[str, float], bool, dict[str, Any]]:
        """Step simulation using dog actions and internal sheep dynamics."""

        prev_center = centroid(self.sheep_pos)
        fallback_used = 0

        for i in range(self.config.n_sheep):
            u_sheep = reynolds_sheep_control(i, self.sheep_pos, self.sheep_vel, self.dog_pos, self.config.u_max_sheep)
            self.sheep_pos[i], self.sheep_vel[i] = semi_implicit_euler(
                self.sheep_pos[i], self.sheep_vel[i], u_sheep, self.config.dt, self.config.drag, self.config.u_max_sheep
            )
            self.sheep_heading[i] = float(np.arctan2(self.sheep_vel[i, 1], self.sheep_vel[i, 0] + 1e-8))

        for j in range(self.config.n_dogs):
            dog_id = f"dog_{j}"
            raw = np.array(action_dict.get(dog_id, np.zeros(2)), dtype=float)
            baseline = self._baseline_dog_action(j)
            others = np.delete(self.dog_pos, j, axis=0)
            safe, fb = self.shield.project_action(dog_id, raw, self.dog_pos[j], self.dog_vel[j], self.config.dt, others, baseline)
            fallback_used += int(fb)
            self.dog_pos[j], self.dog_vel[j] = semi_implicit_euler(
                self.dog_pos[j], self.dog_vel[j], safe, self.config.dt, self.config.drag, self.config.u_max_dog
            )

        self.step_count += 1
        collisions = self._collisions()
        enc = self._sheep_in_goal()
        new_center = centroid(self.sheep_pos)
        goal_progress = self._goal_progress(prev_center, new_center)
        energy = float(sum(np.linalg.norm(np.array(action_dict.get(f"dog_{j}", np.zeros(2)))) for j in range(self.config.n_dogs)))

        rw = self.config.reward_weights
        team_reward = rw.w_enc * enc + rw.w_goal * goal_progress - rw.w_coll * collisions - rw.w_energy * energy - rw.w_comm
        rewards = {f"dog_{j}": team_reward for j in range(self.config.n_dogs)}
        rewards.update({f"sheep_{i}": 0.0 for i in range(self.config.n_sheep)})

        done = bool(enc >= 0.95 or self.step_count >= self.config.max_steps)
        info = {
            "collisions": collisions,
            "enclosure_fraction": enc,
            "goal_progress": goal_progress,
            "messages": 1,
            "energy": energy,
            "fallback_used": fallback_used,
            "intent": self.last_intent,
        }
        return self._get_obs(), rewards, done, info

    def get_state(self) -> dict[str, Any]:
        """Return serializable environment state for reproducibility."""

        return {
            "config": asdict(self.config),
            "step_count": self.step_count,
            "sheep_pos": self.sheep_pos.copy(),
            "sheep_vel": self.sheep_vel.copy(),
            "sheep_heading": self.sheep_heading.copy(),
            "dog_pos": self.dog_pos.copy(),
            "dog_vel": self.dog_vel.copy(),
            "last_intent": self.last_intent,
            "rng_state": self.rng.bit_generator.state,
        }

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore internal state exactly from a state snapshot."""

        self.step_count = int(state["step_count"])
        self.sheep_pos = np.array(state["sheep_pos"], dtype=float)
        self.sheep_vel = np.array(state["sheep_vel"], dtype=float)
        self.sheep_heading = np.array(state["sheep_heading"], dtype=float)
        self.dog_pos = np.array(state["dog_pos"], dtype=float)
        self.dog_vel = np.array(state["dog_vel"], dtype=float)
        self.last_intent = str(state["last_intent"])
        self.rng = np.random.default_rng()
        self.rng.bit_generator.state = state["rng_state"]

    def render(self, mode: str = "human") -> np.ndarray | None:
        """Render a simple top-down visualization as RGB array or interactive plot."""

        import matplotlib.pyplot as plt

        w, h = self.config.map_size
        fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
        ax.set_xlim(0, w)
        ax.set_ylim(0, h)
        ax.set_aspect("equal")
        ax.scatter(self.sheep_pos[:, 0], self.sheep_pos[:, 1], c="white", edgecolors="black", s=45)
        ax.scatter(self.dog_pos[:, 0], self.dog_pos[:, 1], c="black", s=55)
        goal = np.array(self.config.goal)
        gr = self.config.goal_radius
        rect = plt.Rectangle((goal[0] - gr, goal[1] - gr), 2 * gr, 2 * gr, fill=False, color="green", linewidth=2)
        ax.add_patch(rect)
        for poly in self.config.obstacles:
            patch = plt.Polygon(poly, color="steelblue", alpha=0.4)
            ax.add_patch(patch)
        ax.set_title(f"step={self.step_count} intent={self.last_intent}")
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        if mode == "human":
            plt.show(block=False)
            plt.pause(0.001)
            plt.close(fig)
            return None
        plt.close(fig)
        return img
