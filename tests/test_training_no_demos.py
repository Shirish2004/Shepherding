"""Training tests ensuring scratch RL and BC synthetic fallback work without demos."""

from pathlib import Path

import numpy as np

from train.bc_pretrain import collect_synthetic_dataset
from train.mappo_train import run_training


def test_synthetic_bc_dataset_collection() -> None:
    obs, acts = collect_synthetic_dataset(seed=5, samples=64)
    assert obs.shape[0] == 64
    assert acts.shape[0] == 64
    assert obs.ndim == 2 and acts.ndim == 2


def test_scratch_training_runs_without_demo_files(tmp_path: Path) -> None:
    hist = run_training(iters=2, steps_per_iter=20, seed=4, gamma=0.99, lr=1e-3, exploration=0.3, use_curriculum=False)
    assert len(hist["iter"]) == 2
    assert len(hist["episode_reward"]) == 2
