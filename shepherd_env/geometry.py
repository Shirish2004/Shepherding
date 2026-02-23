"""Geometry helpers for collisions, ray intersections, and obstacle inflation."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def norm(vec: np.ndarray) -> float:
    """Return Euclidean norm as float."""

    return float(np.linalg.norm(vec))


def clip_norm(vec: np.ndarray, max_norm: float) -> np.ndarray:
    """Clip vector magnitude to max_norm."""

    n = norm(vec)
    if n <= max_norm or n == 0.0:
        return vec
    return vec * (max_norm / n)


def pairwise_dist_ok(points: np.ndarray, d_min: float) -> bool:
    """Check minimum pairwise distance for a set of points."""

    if len(points) <= 1:
        return True
    for i in range(len(points)):
        delta = points[i + 1 :] - points[i]
        if np.any(np.linalg.norm(delta, axis=1) < d_min):
            return False
    return True


def point_in_polygon(point: np.ndarray, polygon: list[tuple[float, float]]) -> bool:
    """Return True when point lies inside polygon using ray casting."""

    x, y = point
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            x_int = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-12) + x1
            if x < x_int:
                inside = not inside
    return inside


def ray_circle_intersection(origin: np.ndarray, direction: np.ndarray, center: np.ndarray, radius: float) -> float | None:
    """Return nearest positive ray-circle intersection distance if any."""

    oc = origin - center
    b = 2.0 * np.dot(direction, oc)
    c = np.dot(oc, oc) - radius * radius
    disc = b * b - 4.0 * c
    if disc < 0.0:
        return None
    sqrt_disc = math.sqrt(disc)
    t1 = (-b - sqrt_disc) / 2.0
    t2 = (-b + sqrt_disc) / 2.0
    candidates = [t for t in (t1, t2) if t >= 0.0]
    if not candidates:
        return None
    return min(candidates)


def ray_segment_intersection(origin: np.ndarray, direction: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> float | None:
    """Return ray distance to segment intersection if any."""

    v1 = origin - p1
    v2 = p2 - p1
    cross = direction[0] * v2[1] - direction[1] * v2[0]
    if abs(cross) < 1e-12:
        return None
    t = (v2[0] * v1[1] - v2[1] * v1[0]) / cross
    u = (direction[0] * v1[1] - direction[1] * v1[0]) / cross
    if t >= 0.0 and 0.0 <= u <= 1.0:
        return t
    return None


def ray_polygon_intersection(origin: np.ndarray, direction: np.ndarray, polygon: list[tuple[float, float]]) -> float | None:
    """Return nearest ray distance to polygon boundary intersection."""

    poly = [np.array(p, dtype=float) for p in polygon]
    best = None
    for i in range(len(poly)):
        t = ray_segment_intersection(origin, direction, poly[i], poly[(i + 1) % len(poly)])
        if t is None:
            continue
        if best is None or t < best:
            best = t
    return best


def centroid(points: Iterable[np.ndarray]) -> np.ndarray:
    """Compute centroid of a set of vectors."""

    arr = np.array(list(points), dtype=float)
    if len(arr) == 0:
        return np.zeros(2)
    return arr.mean(axis=0)
