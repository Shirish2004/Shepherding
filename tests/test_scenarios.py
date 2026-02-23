"""Integration scenarios covering single and multi-agent configurations."""

import numpy as np

from shepherd_env import EnvConfig, ShepherdEnv


def rollout(env: ShepherdEnv, steps: int = 20) -> dict:
    obs = env.reset(seed=3)
    info = {}
    for _ in range(steps):
        act = {f"dog_{j}": env._baseline_dog_action(j) for j in range(env.config.n_dogs)}
        obs, rewards, done, info = env.step(act)
        if done:
            break
    return {"obs": obs, "info": info}


def test_single_sheep_single_dog_runs() -> None:
    cfg = EnvConfig(n_sheep=1, n_dogs=1, sheep_fov_deg=60.0, spawn_radius=0.5)
    env = ShepherdEnv(cfg)
    out = rollout(env)
    assert "dog_0" in out["obs"] and "sheep_0" in out["obs"]


def test_multi_sheep_single_dog_runs() -> None:
    cfg = EnvConfig(n_sheep=5, n_dogs=1, sheep_fov_deg=90.0)
    env = ShepherdEnv(cfg)
    out = rollout(env)
    assert out["info"]["energy"] >= 0.0


def test_multi_sheep_multi_dog_with_obstacles_runs() -> None:
    obstacles = [[(14.0, 14.0), (16.0, 14.0), (16.0, 16.0), (14.0, 16.0)]]
    cfg = EnvConfig(n_sheep=10, n_dogs=4, sheep_fov_deg=120.0, obstacles=obstacles)
    env = ShepherdEnv(cfg)
    out = rollout(env)
    assert "intent" in out["info"]


def test_visibility_ablation_fov_changes_observability() -> None:
    cfg60 = EnvConfig(n_sheep=3, n_dogs=1, sheep_fov_deg=60.0, sheep_n_rays=36, sheep_lidar_sigma=0.0)
    cfg120 = EnvConfig(n_sheep=3, n_dogs=1, sheep_fov_deg=120.0, sheep_n_rays=36, sheep_lidar_sigma=0.0)
    env60 = ShepherdEnv(cfg60)
    env120 = ShepherdEnv(cfg120)
    o60 = env60.reset(seed=42)
    o120 = env120.reset(seed=42)
    vis60 = np.sum(o60["sheep_0"]["lidar"][:, 1] >= 0)
    vis120 = np.sum(o120["sheep_0"]["lidar"][:, 1] >= 0)
    assert vis120 >= vis60
