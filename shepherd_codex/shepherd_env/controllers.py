"""StringNet-style controller wrappers (seeking, enclosing, herding)."""

from __future__ import annotations

import numpy as np


def sig_alpha(x: np.ndarray, alpha: float) -> np.ndarray:
    """Finite-time smooth sign used in StringNet laws."""
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.power(np.abs(x), alpha)


def omega(u: np.ndarray, ubar: float) -> np.ndarray:
    """Saturation map Ω(u) to bounded acceleration ball."""
    n = np.linalg.norm(u)
    if n <= ubar or n == 0:
        return u
    return (ubar / n) * u


def _desired_point(j: int, formation_params: dict, n_d: int) -> np.ndarray:
    center = np.asarray(formation_params["center"], dtype=float)
    phi0 = float(formation_params.get("phi", 0.0))
    rad = float(formation_params.get("radius", 1.5))
    if n_d == 1:
        ang = phi0
    else:
        ang = phi0 + np.pi / 2 + np.pi * (j) / (n_d - 1)
    return center + rad * np.array([np.cos(ang), np.sin(ang)])


def _base_control(def_id: int, formation_params: dict, state: dict, config: dict, phase: str) -> np.ndarray:
    dogs = np.asarray(state["dog_pos"], dtype=float)
    dog_vel = np.asarray(state["dog_vel"], dtype=float)
    n_d = dogs.shape[0]
    rd = dogs[def_id]
    vd = dog_vel[def_id]
    xi = _desired_point(def_id, formation_params, n_d)
    # Eq-like structure from (13)-(22): acceleration feedforward + Ω(h1)+Ω(h2)
    k1 = float(config.get("k1", 2.0))
    k2 = float(config.get("k2", 1.5))
    alpha1 = float(config.get("alpha1", 0.7))
    alpha2 = float(config.get("alpha2", 0.8))
    cd = float(config.get("C_D", 0.1))

    phase_gain = {"seek": 1.0, "enclose": 1.2, "herd": 1.4}.get(phase, 1.0)
    h1 = -phase_gain * k1 * sig_alpha(rd - xi, alpha1)
    h2 = -k2 * sig_alpha(vd, alpha2) + cd * np.linalg.norm(vd) * vd
    u_col = np.zeros(2)
    for j in range(n_d):
        if j == def_id:
            continue
        diff = rd - dogs[j]
        d = np.linalg.norm(diff)
        if d < 0.6 and d > 1e-6:
            u_col += 0.15 * diff / (d**2)
    u = h1 + h2 + u_col
    return omega(u, float(config["ubar_d"]))


def apply_seeking_controller(def_id: int, formation_params: dict, state: dict, config: dict) -> np.ndarray:
    return _base_control(def_id, formation_params, state, config, "seek")


def apply_enclosing_controller(def_id: int, formation_params: dict, state: dict, config: dict) -> np.ndarray:
    return _base_control(def_id, formation_params, state, config, "enclose")


def apply_herding_controller(def_id: int, formation_params: dict, state: dict, config: dict) -> np.ndarray:
    return _base_control(def_id, formation_params, state, config, "herd")
