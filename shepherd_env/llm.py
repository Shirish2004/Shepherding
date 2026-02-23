"""LLM planner interfaces with deterministic mock implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .geometry import centroid


class LLMPlanner(Protocol):
    """Planner protocol returning intent token and confidence."""

    def plan(self, tokens: dict) -> tuple[str, float]:
        """Return symbolic intent token and confidence."""

    def set_adapter(self, weights: np.ndarray | None) -> None:
        """Set optional adapter weights for downstream embedding."""


@dataclass
class MockLLMPlanner:
    """Deterministic planner that emulates token-level coordination guidance."""

    adapter: np.ndarray | None = None

    def set_adapter(self, weights: np.ndarray | None) -> None:
        """Set simple linear adapter placeholder."""

        self.adapter = weights

    def plan(self, tokens: dict) -> tuple[str, float]:
        """Create intent token by deterministic scene heuristics."""

        spread = float(tokens.get("sheep_spread", 0.0))
        goal = np.array(tokens.get("goal", [0.0, 0.0]), dtype=float)
        center = np.array(tokens.get("sheep_center", [0.0, 0.0]), dtype=float)
        delta = goal - center
        angle = float(np.arctan2(delta[1], delta[0]))
        if spread > 3.0:
            return "SPLIT:LEFT", 0.72
        if np.linalg.norm(delta) < 1.0:
            return "HOLD", 0.88
        return f"ORIENT:{angle:.3f}", 0.91


def scene_tokens(sheep_positions: np.ndarray, sheep_velocities: np.ndarray, goal: tuple[float, float]) -> dict:
    """Build symbolic scene summary tokens from raw states."""

    center = centroid(sheep_positions)
    spread = float(np.max(np.linalg.norm(sheep_positions - center, axis=1))) if len(sheep_positions) else 0.0
    heading = centroid(sheep_velocities)
    split_flag = bool(spread > 2.5)
    return {
        "sheep_center": center.tolist(),
        "sheep_spread": spread,
        "sheep_heading": float(np.arctan2(heading[1], heading[0] + 1e-8)),
        "split_flag": split_flag,
        "goal": [float(goal[0]), float(goal[1])],
    }


INTENT_VOCAB = ["HOLD", "SPLIT:LEFT", "SPLIT:RIGHT", "CLOSE_GAP", "MOVE_TEAM", "ORIENT"]


def intent_to_embedding(intent: str) -> np.ndarray:
    """Map a string intent token into fixed one-hot embedding."""

    prefix = intent.split(":", maxsplit=1)[0]
    vec = np.zeros(len(INTENT_VOCAB), dtype=float)
    if prefix in INTENT_VOCAB:
        vec[INTENT_VOCAB.index(prefix)] = 1.0
    return vec
