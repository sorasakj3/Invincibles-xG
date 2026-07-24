from math import isfinite

from invincibles_xg.features import shot_geometry


def test_distance_decreases_toward_goal() -> None:
    far_distance, _ = shot_geometry([80, 40])
    near_distance, _ = shot_geometry([112, 40])
    assert near_distance < far_distance


def test_angle_is_finite_and_wider_centrally() -> None:
    _, central = shot_geometry([108, 40])
    _, wide = shot_geometry([108, 72])
    assert isfinite(central)
    assert central > wide

