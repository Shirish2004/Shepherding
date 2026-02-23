"""LiDAR sensing tests for hit detection, occlusion, and noise behavior."""

import numpy as np

from shepherd_env.lidar import CLASS_TO_ID, LidarSensor


def test_lidar_detects_dog_hit_front_ray() -> None:
    sensor = LidarSensor(90.0, 9, 10.0, 0.0, 0.0, 0.0, 0.0, 0.2)
    rng = np.random.default_rng(0)
    sheep = np.array([[0.0, 0.0]])
    dogs = np.array([[3.0, 0.0]])
    scan = sensor.scan(np.array([0.0, 0.0]), 0.0, sheep, dogs, [], 0, rng)
    mid = scan[len(scan) // 2]
    assert int(mid[1]) == CLASS_TO_ID["dog"]
    assert 2.7 < mid[0] < 3.0


def test_lidar_occlusion_nearest_object_selected() -> None:
    sensor = LidarSensor(90.0, 9, 10.0, 0.0, 0.0, 0.0, 0.0, 0.2)
    rng = np.random.default_rng(0)
    sheep = np.array([[0.0, 0.0], [5.0, 0.0]])
    dogs = np.array([[2.0, 0.0]])
    scan = sensor.scan(np.array([0.0, 0.0]), 0.0, sheep, dogs, [], 0, rng)
    mid = scan[len(scan) // 2]
    assert int(mid[1]) == CLASS_TO_ID["dog"]
    assert mid[0] < 2.0


def test_lidar_noise_changes_distances() -> None:
    sensor = LidarSensor(90.0, 9, 10.0, 0.1, 0.0, 0.0, 0.0, 0.2)
    sheep = np.array([[0.0, 0.0]])
    dogs = np.array([[3.0, 0.0]])
    scan_a = sensor.scan(np.array([0.0, 0.0]), 0.0, sheep, dogs, [], 0, np.random.default_rng(1))
    scan_b = sensor.scan(np.array([0.0, 0.0]), 0.0, sheep, dogs, [], 0, np.random.default_rng(2))
    assert not np.allclose(scan_a[:, 0], scan_b[:, 0])
