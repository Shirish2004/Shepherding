"""Configuration dataclasses for the shepherding environment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RewardWeights:
    """Reward weight container for multi-term shaping."""

    w_enc: float = 5.0
    w_goal: float = 1.0
    w_coll: float = 50.0
    w_comm: float = 0.01
    w_energy: float = 0.001


@dataclass
class EnvConfig:
    """Main environment configuration with physically motivated defaults."""

    n_sheep: int = 5
    n_dogs: int = 3
    map_size: tuple[float, float] = (30.0, 30.0)
    dt: float = 0.05
    drag: float = 0.1
    u_max_dog: float = 3.0
    u_max_sheep: float = 2.0
    r_agent: float = 0.2
    max_steps: int = 600
    sensing_range: float = 10.0
    k_nearest_sheep: int = 5
    include_sheep_pos: bool = True
    sheep_fov_deg: float = 90.0
    sheep_n_rays: int = 90
    sheep_lidar_range: float = 8.0
    sheep_lidar_sigma: float = 0.01
    sheep_false_negative: float = 0.0
    sheep_false_positive: float = 0.0
    sheep_intensity_noise: float = 0.02
    spawn_center: tuple[float, float] = (10.0, 10.0)
    spawn_radius: float = 1.5
    dog_spawn_mode: str = "periphery"
    dog_spawn_nodes: list[tuple[float, float]] = field(default_factory=list)
    min_pair_distance: float = 0.45
    goal: tuple[float, float] = (24.0, 24.0)
    goal_radius: float = 2.0
    obstacles: list[list[tuple[float, float]]] = field(default_factory=list)
    obstacle_inflation: float = 0.1
    seed: int | None = None
    reward_weights: RewardWeights = field(default_factory=RewardWeights)

    def validate(self) -> None:
        """Validate scalar ranges and model constraints."""

        if not (1 <= self.n_sheep and 1 <= self.n_dogs):
            raise ValueError("n_sheep and n_dogs must be >=1")
        if not (60.0 <= self.sheep_fov_deg <= 120.0):
            raise ValueError("sheep_fov_deg must be in [60, 120]")
        if self.sheep_n_rays < 1:
            raise ValueError("sheep_n_rays must be positive")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")

    @classmethod
    def from_overrides(cls, base: "EnvConfig", overrides: dict[str, Any]) -> "EnvConfig":
        """Create a validated config from an existing config and overrides."""

        data = base.__dict__.copy()
        for key, value in overrides.items():
            data[key] = value
        cfg = cls(**data)
        cfg.validate()
        return cfg
