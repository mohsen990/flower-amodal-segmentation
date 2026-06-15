"""Precision and Recall convergence across checkpoints (4 lines)."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

OUT = os.path.join(os.path.dirname(__file__), "fig_precision_recall.pdf")

iters = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30]

# IoU ≥ 0.50
p50 = [0.8973, 0.9105, 0.9589, 0.9469, 0.9568, 0.9756, 0.9725, 0.9826, 0.9892, 0.9840]
r50 = [0.8872, 0.9357, 0.9321, 0.9522, 0.9426, 0.9627, 0.9805, 0.9713, 0.9706, 0.9759]

# IoU ≥ 0.75
p75 = [0.6841, 0.7894, 0.8588, 0.8870, 0.9017, 0.9463, 0.9168, 0.9560, 0.9644, 0.9667]
r75 = [0.6765, 0.7764, 0.8348, 0.8724, 0.8559, 0.9014, 0.8978, 0.9449, 0.9462, 0.9469]

fig, axes = plt.subplots(1, 2, figsize=(10, 4.0), sharey=False)

styles = {
    "Precision": {"ls": "-",  "marker": "o"},
    "Recall":    {"ls": "--", "marker": "s"},
}
colors_50 = {"Precision": "#2196F3", "Recall": "#03A9F4"}
colors_75 = {"Precision": "#E64A19", "Recall": "#FF7043"}

for ax, (prec, rec), colors, iou_tag, ylim_lo in zip(
        axes,
        [(p50, r50), (p75, r75)],
        [colors_50, colors_75],
        ["IoU ≥ 0.50", "IoU ≥ 0.75"],
        [0.86, 0.62]):

    ax.plot(iters, prec, color=colors["Precision"], linewidth=2,
            linestyle="-", marker="o", markersize=5, label="Precision", zorder=3)
    ax.plot(iters, rec,  color=colors["Recall"],    linewidth=2,
            linestyle="--", marker="s", markersize=5, label="Recall",    zorder=3)

    ax.set_xlabel("Training Iterations (×1000)", fontsize=10)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_title(f"Precision & Recall @ Mask {iou_tag}", fontsize=11, fontweight="bold")
    ax.set_xticks(iters)
    ax.set_ylim(ylim_lo, 1.01)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9, loc="lower right")

fig.suptitle("Precision and Recall Convergence — FlowerAmodal v3",
             fontsize=10, y=1.01)
plt.tight_layout()
plt.savefig(OUT, dpi=300, bbox_inches="tight")
print(f"Saved {OUT}")
