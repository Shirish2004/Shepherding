"""Utilities for adapter training data encoding and optimization."""

from __future__ import annotations

import torch


def scene_to_tensor(scene_tokens: dict) -> torch.Tensor:
    """Vectorize symbolic scene tokens to a fixed-size torch tensor."""
    return torch.tensor(
        [
            float(scene_tokens["ACoM"][0]),
            float(scene_tokens["ACoM"][1]),
            float(scene_tokens["sheep_spread"]),
            float(scene_tokens["largest_cluster_dist"]),
            float(scene_tokens["escape_prob_est"]),
            float(scene_tokens["obstacle_density_nearby"]),
        ],
        dtype=torch.float32,
    )


def intent_to_idx(intent: dict, vocab: list[str]) -> int:
    tok = intent.get("intent_token", "tighten_net")
    return vocab.index(tok)
