"""Shared plot colours and helpers — use this from every role's plotting script."""
from pathlib import Path

import matplotlib.pyplot as plt

# Fixed palette so everyone's figures match in the report / slides.
COLOURS = {
    "baseline": "#1f4e79",
    "cpu": "#2e75b6",
    "gpu": "#c45911",
    "amp": "#ed7d31",
    "ddp": "#548235",
    "loader": "#7030a0",
    "io_edf": "#a50021",
    "io_npz": "#2e75b6",
    "io_hdf5": "#548235",
    "dask": "#00a3a1",
}

STAGE_COLOURS = {
    "W": "#4d4d4d",
    "N1": "#e69138",
    "N2": "#6fa8dc",
    "N3": "#3d85c6",
    "REM": "#a64d79",
}


def apply_style():
    plt.rcParams.update({
        "figure.figsize": (6.5, 4.0),
        "figure.dpi": 120,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
    })


def savefig(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    print(f"Wrote {path}")
