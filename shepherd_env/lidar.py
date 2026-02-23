"""LiDAR sensor simulation for sheep-centric perception."""

from __future__ import annotations

import math
import numpy as np

from .geometry import ray_circle_intersection, ray_polygon_intersection

CLASS_TO_ID = {"none": -1, "sheep": 0, "dog": 1, "obstacle": 2}


class LidarSensor:
    """Configurable 2D ray-casting LiDAR with occlusion and noise."""

    def __init__(
        self,
        fov_deg: float,
        n_rays: int,
        max_range: float,
        sigma_range: float,
        false_negative: float,
        false_positive: float,
        intensity_noise: float,
        agent_radius: float,
    ) -> None:
        if not (60.0 <= fov_deg <= 120.0):
            raise ValueError("LiDAR FOV must be in [60, 120] degrees")
        self.fov_deg = fov_deg
        self.n_rays = n_rays
        self.max_range = max_range
        self.sigma_range = sigma_range
        self.false_negative = false_negative
        self.false_positive = false_positive
        self.intensity_noise = intensity_noise
        self.agent_radius = agent_radius

    def scan(
        self,
        origin: np.ndarray,
        heading: float,
        sheep_positions: np.ndarray,
        dog_positions: np.ndarray,
        obstacles: list[list[tuple[float, float]]],
        self_index: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Return LiDAR tuples (distance, class_id, bearing, intensity) for each ray."""

        half = math.radians(self.fov_deg) / 2.0
        bearings = np.linspace(-half, half, self.n_rays)
        out = np.zeros((self.n_rays, 4), dtype=float)
        for i, b in enumerate(bearings):
            ray_angle = heading + b
            direction = np.array([math.cos(ray_angle), math.sin(ray_angle)], dtype=float)
            best_dist = self.max_range
            best_cls = CLASS_TO_ID["none"]

            for s_idx, pos in enumerate(sheep_positions):
                if s_idx == self_index:
                    continue
                t = ray_circle_intersection(origin, direction, pos, self.agent_radius)
                if t is not None and t < best_dist and t <= self.max_range:
                    best_dist = t
                    best_cls = CLASS_TO_ID["sheep"]

            for pos in dog_positions:
                t = ray_circle_intersection(origin, direction, pos, self.agent_radius)
                if t is not None and t < best_dist and t <= self.max_range:
                    best_dist = t
                    best_cls = CLASS_TO_ID["dog"]

            for poly in obstacles:
                t = ray_polygon_intersection(origin, direction, poly)
                if t is not None and t < best_dist and t <= self.max_range:
                    best_dist = t
                    best_cls = CLASS_TO_ID["obstacle"]

            if rng.random() < self.false_negative:
                best_cls = CLASS_TO_ID["none"]
                best_dist = self.max_range

            if best_cls == CLASS_TO_ID["none"] and rng.random() < self.false_positive:
                best_cls = int(rng.choice([CLASS_TO_ID["sheep"], CLASS_TO_ID["dog"], CLASS_TO_ID["obstacle"]]))
                best_dist = float(rng.uniform(0.1, self.max_range))

            noisy_dist = float(np.clip(best_dist + rng.normal(0.0, self.sigma_range), 0.0, self.max_range))
            intensity = float(np.clip(1.0 - noisy_dist / self.max_range + rng.normal(0.0, self.intensity_noise), 0.0, 1.0))
            out[i] = np.array([noisy_dist, float(best_cls), float(b), intensity])
        return out
