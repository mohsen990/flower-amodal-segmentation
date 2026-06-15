"""All 4 segmentation tasks AP comparison (amodal, visible, occluded, occluding)."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

OUT = os.path.join(os.path.dirname(__file__), "fig_all_segm_tasks.pdf")

tasks = ["Amodal\nSegm", "Visible\nSegm", "Occluded\nSegm", "Occluding\nSegm"]

# AP (0.5:0.95), AP50, AP75, AR
data = {
    "AP (0.5:0.95)": [85.12, 90.11, 19.61, 19.66],
    "AP50":          [98.35, 98.44, 45.97, 45.51],
    "AP75":          [95.46, 97.23, 13.30, 13.72],
    "AR":            [69.41, 71.84, 33.34, 33.39],
}
colors = ["#4C72B0", "#55A868", "#DD8452", "#C44E52"]

x      = np.arange(len(tasks))
n      = len(data)
width  = 0.18
offsets = np.linspace(-(n-1)/2, (n-1)/2, n) * width

fig, ax = plt.subplots(figsize=(9, 4.8))

for i, (label, vals) in enumerate(data.items()):
    bars = ax.bar(x + offsets[i], vals, width, label=label,
                  color=colors[i], edgecolor="white", linewidth=0.6, zorder=3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{v:.1f}", ha="center", va="bottom",
                fontsize=6, fontweight="bold", rotation=0)

# highlight the gap between easy tasks (amodal/visible) and hard (occluded/occluding)
ax.axvline(x=1.5, color="#AAAAAA", linestyle=":", linewidth=1.5)
ax.text(0.75, 107, "Easy", ha="center", fontsize=9, color="#555555", style="italic")
ax.text(2.25, 107, "Hard", ha="center", fontsize=9, color="#555555", style="italic")

ax.set_xticks(x)
ax.set_xticklabels(tasks, fontsize=10.5)
ax.set_ylabel("Score (%)", fontsize=11)
ax.set_ylim(0, 115)
ax.set_yticks(range(0, 101, 20))
ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="upper right", fontsize=8.5, framealpha=0.9)
ax.set_title("All Segmentation Tasks — COCO Metrics (val, FlowerAmodal v3)",
             fontsize=10, pad=10)

plt.tight_layout()
plt.savefig(OUT, dpi=300, bbox_inches="tight")
print(f"Saved {OUT}")
