from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

PITCH_LENGTH = 120.0
PITCH_WIDTH = 80.0
GOAL_WIDTH = 8.0


def shot_geometry(location: list[float]) -> tuple[float, float]:
    """Return distance to goal centre and visible goal angle in radians."""
    x, y = float(location[0]), float(location[1])
    dx = max(PITCH_LENGTH - x, 1e-6)
    dy = abs(PITCH_WIDTH / 2 - y)
    distance = math.hypot(dx, dy)
    left = math.atan2(PITCH_WIDTH / 2 + GOAL_WIDTH / 2 - y, dx)
    right = math.atan2(PITCH_WIDTH / 2 - GOAL_WIDTH / 2 - y, dx)
    angle = abs(left - right)
    if angle > math.pi:
        angle = 2 * math.pi - angle
    return distance, angle


def _shot_row(event: dict, match_id: str) -> dict:
    shot = event.get("shot", {})
    body_part = shot.get("body_part", {}).get("name", "Unknown")
    technique = shot.get("technique", {}).get("name", "Unknown")
    play_pattern = event.get("play_pattern", {}).get("name", "Unknown")
    distance, angle = shot_geometry(event["location"])
    outcome = shot.get("outcome", {}).get("name", "")
    return {
        "match_id": str(match_id),
        "event_id": event["id"],
        "team": event["team"]["name"],
        "player": event.get("player", {}).get("name", "Unknown"),
        "minute": event.get("minute", 0),
        "x": float(event["location"][0]),
        "y": float(event["location"][1]),
        "distance": distance,
        "angle": angle,
        "under_pressure": int(event.get("under_pressure", False)),
        "first_time": int(shot.get("first_time", False)),
        "header": int(body_part == "Head"),
        "open_play": int(play_pattern == "Regular Play"),
        "technique": technique,
        "body_part": body_part,
        "statsbomb_xg": float(shot.get("statsbomb_xg", 0.0)),
        "goal": int(outcome == "Goal"),
    }


def build_shot_table(events_dir: str | Path) -> pd.DataFrame:
    rows: list[dict] = []
    for path in sorted(Path(events_dir).glob("*.json")):
        events = json.loads(path.read_text())
        rows.extend(
            _shot_row(event, path.stem)
            for event in events
            if event.get("type", {}).get("name") == "Shot" and event.get("location")
        )
    if not rows:
        raise ValueError(f"No StatsBomb shot events found in {events_dir}")
    return pd.DataFrame(rows).sort_values(["match_id", "minute", "event_id"])

