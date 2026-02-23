"""Damped double-integrator dynamics and behavior models."""

from __future__ import annotations

import numpy as np

from .geometry import clip_norm


def semi_implicit_euler(pos: np.ndarray, vel: np.ndarray, acc_cmd: np.ndarray, dt: float, drag: float, u_max: float) -> tuple[np.ndarray, np.ndarray]:
    """Advance damped double integrator using semi-implicit Euler."""

    acc = clip_norm(acc_cmd, u_max)
    damped = acc - drag * np.linalg.norm(vel) * vel
    new_vel = vel + dt * damped
    new_pos = pos + dt * new_vel
    return new_pos, new_vel


def reynolds_sheep_control(
    idx: int,
    sheep_pos: np.ndarray,
    sheep_vel: np.ndarray,
    dogs_pos: np.ndarray,
    u_max: float,
) -> np.ndarray:
    """Compute bounded sheep acceleration via separation-alignment-cohesion and dog repulsion."""

    p_i = sheep_pos[idx]
    v_i = sheep_vel[idx]
    others = np.delete(sheep_pos, idx, axis=0)
    others_vel = np.delete(sheep_vel, idx, axis=0)

    separation = np.zeros(2)
    if len(others) > 0:
        d = others - p_i
        r = np.linalg.norm(d, axis=1, keepdims=True) + 1e-6
        separation = -np.sum(d / (r * r), axis=0)
        cohesion = others.mean(axis=0) - p_i
        alignment = others_vel.mean(axis=0) - v_i
    else:
        cohesion = np.zeros(2)
        alignment = -0.1 * v_i

    dog_repulsion = np.zeros(2)
    if len(dogs_pos) > 0:
        d = dogs_pos - p_i
        r = np.linalg.norm(d, axis=1, keepdims=True) + 1e-6
        dog_repulsion = -2.5 * np.sum(d / (r * r), axis=0)

    control = 1.6 * separation + 0.4 * cohesion + 0.5 * alignment + dog_repulsion
    return clip_norm(control, u_max)


def chipade_panagou_enclosing_accel(
    dog_pos: np.ndarray,
    dog_vel: np.ndarray,
    assigned_goal: np.ndarray,
    assigned_goal_vel: np.ndarray,
    k1: float = 1.2,
    k2: float = 1.0,
    drag: float = 0.1,
) -> np.ndarray:
    """Compute enclosing acceleration inspired by equations 19-22 in Chipade-Panagou."""

    term_pos = -k1 * (dog_pos - assigned_goal)
    term_vel = -k2 * (dog_vel - assigned_goal_vel)
    return term_pos + term_vel + drag * np.linalg.norm(dog_vel) * dog_vel
