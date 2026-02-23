"""Spawn samplers for sheep clusters and dogs with deterministic RNG support."""

from __future__ import annotations

import math
import numpy as np

from .geometry import pairwise_dist_ok


def sample_points_in_disc(rng: np.random.Generator, center: np.ndarray, radius: float, n: int, d_min: float, max_tries: int = 5000) -> np.ndarray:
    """Sample points in a disc with minimum pairwise distance."""

    points: list[np.ndarray] = []
    tries = 0
    while len(points) < n and tries < max_tries:
        tries += 1
        rho = radius * math.sqrt(float(rng.random()))
        theta = 2.0 * math.pi * float(rng.random())
        candidate = center + np.array([rho * math.cos(theta), rho * math.sin(theta)])
        if points:
            existing = np.array(points)
            if np.any(np.linalg.norm(existing - candidate, axis=1) < d_min):
                continue
        points.append(candidate)
    if len(points) != n:
        raise RuntimeError("Unable to sample disc points under d_min constraints")
    arr = np.array(points, dtype=float)
    if not pairwise_dist_ok(arr, d_min):
        raise RuntimeError("Invalid spawn sampling output")
    return arr


def sample_dogs(
    rng: np.random.Generator,
    n: int,
    map_size: tuple[float, float],
    mode: str,
    nodes: list[tuple[float, float]],
    center: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Sample dog starting positions based on periphery or nodes."""

    w, h = map_size
    if mode == "nodes" and nodes:
        chosen = [np.array(nodes[i % len(nodes)], dtype=float) for i in range(n)]
        jitter = rng.normal(0.0, 0.1, size=(n, 2))
        return np.array(chosen) + jitter
    if mode == "near_nodes" and nodes:
        idx = rng.integers(0, len(nodes), size=n)
        base = np.array([nodes[int(i)] for i in idx], dtype=float)
        jitter = rng.normal(0.0, 0.5, size=(n, 2))
        return base + jitter
    perimeter = []
    for i in range(n):
        t = 2.0 * math.pi * i / n
        ring_r = min(w, h) * 0.42
        perimeter.append(center + np.array([ring_r * math.cos(t), ring_r * math.sin(t)]))
    return np.array(perimeter, dtype=float)
