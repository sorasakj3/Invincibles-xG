from __future__ import annotations

import argparse
import json
from pathlib import Path

from .features import build_shot_table
from .model import evaluate_grouped, fit_model
from .visualise import shot_map


def run(events_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    shots = build_shot_table(events_dir)
    evaluation, out_of_fold = evaluate_grouped(shots)
    shots["model_xg"] = out_of_fold
    model = fit_model(shots)
    shots["fitted_xg"] = model.predict_proba(
        shots[["distance", "angle", "under_pressure", "first_time", "header", "open_play", "technique", "body_part"]]
    )[:, 1]
    shots.to_csv(output_dir / "shot_predictions.csv", index=False)
    (output_dir / "metrics.json").write_text(
        json.dumps(evaluation.as_dict(), indent=2) + "\n"
    )
    shot_map(shots, output_dir / "shot_map.png")
    print(json.dumps(evaluation.as_dict(), indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("events_dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    run(args.events_dir, args.output)


if __name__ == "__main__":
    main()

