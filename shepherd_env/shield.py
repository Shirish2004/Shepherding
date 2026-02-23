"""Safety shield with projection and baseline fallback logic."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import clip_norm


@dataclass
class ShieldState:
    """Mutable counters for action violation tracking and fallback windows."""

    violations: dict[str, int] = field(default_factory=dict)
    fallback_steps: dict[str, int] = field(default_factory=dict)


class SafetyShield:
    """Project unsafe actions and switch to baseline when repeated violations happen."""

    def __init__(self, u_max: float, r_agent: float, trigger_count: int = 4, fallback_horizon: int = 8) -> None:
        self.u_max = u_max
        self.r_agent = r_agent
        self.trigger_count = trigger_count
        self.fallback_horizon = fallback_horizon
        self.state = ShieldState()

    def _unsafe(self, dog_pos: np.ndarray, candidate_pos: np.ndarray, other_positions: np.ndarray) -> bool:
        if len(other_positions) == 0:
            return False
        min_d = np.min(np.linalg.norm(other_positions - candidate_pos, axis=1))
        return bool(min_d < 2.0 * self.r_agent)

    def project_action(
        self,
        agent_id: str,
        action: np.ndarray,
        dog_pos: np.ndarray,
        dog_vel: np.ndarray,
        dt: float,
        other_positions: np.ndarray,
        baseline_action: np.ndarray,
    ) -> tuple[np.ndarray, bool]:
        """Return safe action and whether fallback controller was used."""

        action = clip_norm(action.astype(float), self.u_max)
        if self.state.fallback_steps.get(agent_id, 0) > 0:
            self.state.fallback_steps[agent_id] -= 1
            return clip_norm(baseline_action, self.u_max), True

        predicted = dog_pos + dt * (dog_vel + dt * action)
        if self._unsafe(dog_pos, predicted, other_positions):
            self.state.violations[agent_id] = self.state.violations.get(agent_id, 0) + 1
            safe = np.zeros(2, dtype=float)
            if self.state.violations[agent_id] >= self.trigger_count:
                self.state.fallback_steps[agent_id] = self.fallback_horizon
                self.state.violations[agent_id] = 0
                return clip_norm(baseline_action, self.u_max), True
            return safe, False

        self.state.violations[agent_id] = 0
        return action, False
