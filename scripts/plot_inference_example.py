"""Render the README's conductivity comparison from its published curve data.

Run from any directory with NumPy and Matplotlib installed. No heat solver,
sampling runtime, or original chain archives are needed to redraw the figure.
"""

from pathlib import Path
import csv
import hashlib
import json
import os
import tempfile

import numpy as np

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "heat-readme-matplotlib")
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ASSETS = Path(__file__).resolve().parents[1] / "docs/_static/figures"


def main():
    data_path = ASSETS / "conductivity-slices.csv"
    metadata = json.loads((ASSETS / "conductivity-slices.json").read_text())
    assert hashlib.sha256(data_path.read_bytes()).hexdigest() == metadata["csv_sha256"]
    with data_path.open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 258
    blue, orange, graphite = "#2868A2", "#D97932", "#3C4247"
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "text.color": graphite,
            "axes.labelcolor": graphite,
            "xtick.color": graphite,
            "ytick.color": graphite,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "axes.edgecolor": "#9AA0A5",
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "svg.fonttype": "path",
            "pdf.fonttype": 42,
            "svg.hashsalt": "heat-inverse-readme",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), sharey=True)
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.18, top=0.80, wspace=0.15)
    handles = [
        Line2D([], [], color=blue, lw=2.4, label="Sampling"),
        Line2D([], [], color=orange, lw=1.7, ls=(0, (5, 3)), label="Optimization"),
        Line2D([], [], color=graphite, lw=1.25, ls=(0, (1.5, 2.5)), label="Truth"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.53, 0.99),
        ncol=3,
        frameon=False,
        handlelength=2.7,
        columnspacing=2.4,
    )
    for ax, direction, label in zip(
        axes, ("horizontal", "vertical"), ("Horizontal position", "Vertical position")
    ):
        selected = [row for row in rows if row["direction"] == direction]
        assert len(selected) == 129
        values = {
            key: np.array([float(row[key]) for row in selected])
            for key in selected[0]
            if key != "direction"
        }
        x = values["position"]
        assert np.all(np.diff(x) > 0)
        assert np.all(values["lower95"] <= values["lower68"])
        assert np.all(values["lower68"] <= values["upper68"])
        assert np.all(values["upper68"] <= values["upper95"])
        ax.fill_between(
            x, values["lower95"], values["lower68"], color=blue, alpha=0.13, lw=0
        )
        ax.fill_between(
            x, values["upper68"], values["upper95"], color=blue, alpha=0.13, lw=0
        )
        ax.fill_between(
            x, values["lower68"], values["upper68"], color=blue, alpha=0.29, lw=0
        )
        ax.plot(
            x, values["truth"], color=graphite, lw=1.25, ls=(0, (1.5, 2.5)), zorder=3
        )
        ax.plot(x, values["sampling_mean"], color=blue, lw=2.4, zorder=4)
        ax.plot(
            x, values["optimization"], color=orange, lw=1.7, ls=(0, (5, 3)), zorder=5
        )
        ax.set(xlim=(0, 1), ylim=(1.10, 1.80), xlabel=label)
        ax.set_xticks([0, 0.5, 1], labels=["0", "0.5", "1"])
        ax.set_yticks([1.2, 1.4, 1.6, 1.8])
        ax.tick_params(length=3, pad=5)
        for key in ("lower95", "upper95", "truth", "optimization", "sampling_mean"):
            assert values[key].min() > 1.10 and values[key].max() < 1.80
    axes[0].set_ylabel("Conductivity", labelpad=10)
    for extension in ("svg", "png", "pdf"):
        metadata = (
            {"Date": None}
            if extension == "svg"
            else ({"CreationDate": None, "ModDate": None} if extension == "pdf" else {})
        )
        fig.savefig(
            ASSETS / f"conductivity-inference.{extension}",
            dpi=200,
            facecolor="white",
            metadata=metadata,
        )
    plt.close(fig)
    print(
        "Rendered conductivity-inference.svg, .png and .pdf from 258 verified curve points."
    )


if __name__ == "__main__":
    main()
