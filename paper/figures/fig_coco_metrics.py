"""Generate COCO metrics bar chart for the flower amodal segmentation paper."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

OUT = os.path.join(os.path.dirname(__file__), "fig_coco_metrics.pdf")

# ── data ──────────────────────────────────────────────────────────────────────
tasks  = ["Amodal\nSegm", "Visible\nSegm", "BBox"]
# AP (0.5:0.95), AP50, AP75, AR
metrics = {
    "AP (0.5:0.95)": [85.12, 90.11, 81.83],
    "AP50":          [98.35, 98.44, 98.40],
    "AP75":          [95.46, 97.23, 95.34],
    "AR":            [69.41, 71.84, 85.90],
}
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

# ── layout ────────────────────────────────────────────────────────────────────
x      = np.arange(len(tasks))
n      = len(metrics)
width  = 0.18
offsets = np.linspace(-(n-1)/2, (n-1)/2, n) * width

fig, ax = plt.subplots(figsize=(7, 4.5))

bars_list = []
for i, (label, vals) in enumerate(metrics.items()):
    bars = ax.bar(x + offsets[i], vals, width, label=label,
                  color=colors[i], edgecolor="white", linewidth=0.6, zorder=3)
    bars_list.append(bars)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{v:.1f}", ha="center", va="bottom", fontsize=6.5, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(tasks, fontsize=11)
ax.set_ylabel("Score (%)", fontsize=11)
ax.set_ylim(0, 112)
ax.set_yticks(range(0, 101, 20))
ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)
ax.set_title("COCO Evaluation Metrics — Flower Amodal Segmentation (v3, 30k iter)",
             fontsize=9.5, pad=10)

plt.tight_layout()
plt.savefig(OUT, dpi=300, bbox_inches="tight")
print(f"Saved {OUT}")
