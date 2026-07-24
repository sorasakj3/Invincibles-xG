from __future__ import annotations

import urllib.request
from pathlib import Path

MATCH_IDS = [
    3749448, 3749246, 3749552, 3749079, 3749276, 3749068,
    3913152, 3913105, 3913177, 3913083, 3913142, 3912598,
]
BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data/events"


def main() -> None:
    target = Path("data/raw/events")
    target.mkdir(parents=True, exist_ok=True)
    for match_id in MATCH_IDS:
        destination = target / f"{match_id}.json"
        if not destination.exists():
            print(f"Downloading {match_id}")
            urllib.request.urlretrieve(f"{BASE}/{match_id}.json", destination)


if __name__ == "__main__":
    main()

