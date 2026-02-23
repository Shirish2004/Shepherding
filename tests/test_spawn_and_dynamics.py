"""Spawn and dynamics tests for geometric constraints and stability."""

import numpy as np

from shepherd_env.dynamics import semi_implicit_euler
from shepherd_env.spawn import sample_points_in_disc


def test_spawn_points_respect_min_distance() -> None:
    rng = np.random.default_rng(12)
    pts = sample_points_in_disc(rng, np.array([0.0, 0.0]), 2.0, 10, 0.3)
    for i in range(len(pts)):
        d = np.linalg.norm(pts[i + 1 :] - pts[i], axis=1)
        if len(d) > 0:
            assert np.min(d) >= 0.3


def test_dynamics_remain_bounded_under_drag() -> None:
    pos = np.zeros(2)
    vel = np.zeros(2)
    for _ in range(500):
        pos, vel = semi_implicit_euler(pos, vel, np.array([3.0, 0.0]), 0.05, 0.1, 3.0)
    assert np.linalg.norm(vel) < 6.0
