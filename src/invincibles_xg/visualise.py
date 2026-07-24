from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Arc, Rectangle


def _pitch(ax) -> None:
    ax.add_patch(Rectangle((0, 0), 120, 80, fill=False, color="#d7ddd5", lw=1.2))
    ax.plot([60, 60], [0, 80], color="#d7ddd5", lw=1)
    ax.add_patch(Rectangle((102, 18), 18, 44, fill=False, color="#d7ddd5"))
    ax.add_patch(Rectangle((114, 30), 6, 20, fill=False, color="#d7ddd5"))
    ax.add_patch(Arc((108, 40), 20, 20, theta1=127, theta2=233, color="#d7ddd5"))
    ax.set(xlim=(60, 121), ylim=(80, 0), aspect="equal")
    ax.axis("off")


def shot_map(shots: pd.DataFrame, output: str | Path, team_filter: str = "Arsenal") -> None:
    selected = shots[shots["team"].str.contains(team_filter, case=False, na=False)]
    if selected.empty:
        selected = shots
    fig, ax = plt.subplots(figsize=(11, 7), facecolor="#101612")
    ax.set_facecolor("#101612")
    _pitch(ax)
    sizes = 70 + selected["model_xg"] * 900
    colours = selected["goal"].map({0: "#9ba99f", 1: "#ef4035"})
    ax.scatter(
        selected["x"], selected["y"], s=sizes, c=colours, alpha=0.8,
        edgecolors="#f5f2e9", linewidths=0.6
    )
    ax.text(61, 5, "SHOT QUALITY", color="#ef4035", fontsize=11, weight="bold")
    ax.text(
        61, 10, f"{team_filter} · open-data sample", color="#f5f2e9",
        fontsize=19, weight="bold"
    )
    ax.text(
        61, 15, "Marker area = calibrated model xG · red = goal",
        color="#9ba99f", fontsize=10
    )
    fig.tight_layout()
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

