"""Val vs Test F1 gap — like tomato paper Fig. 3."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

OUT = os.path.join(os.path.dirname(__file__), "fig_val_test_gap.pdf")

groups  = ["Val\n(1,516 im.)", "Test\n(1,586 im.)"]
f1_50   = [0.9800, 0.9834]
f1_75   = [0.9567, 0.9548]

x      = np.arange(len(groups))
width  = 0.32

fig, axes = plt.subplots(1, 2, figsize=(8, 4.0))

for ax, vals, iou_tag, ylim_lo, color in zip(
        axes,
        [f1_50, f1_75],
        ["IoU ≥ 0.50", "IoU ≥ 0.75"],
        [0.97, 0.93],
        ["#4C72B0", "#DD8452"]):

    bars = ax.bar(x, vals, width=0.5, color=[color, "#55A868"],
                  edgecolor="white", linewidth=0.8, zorder=3)

    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.0008,
                f"{v:.4f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold")

    # delta annotation
    delta = vals[1] - vals[0]
    sign  = "+" if delta >= 0 else ""
    ax.annotate("", xy=(x[1], vals[1]), xytext=(x[0], vals[0]),
                arrowprops=dict(arrowstyle="<->", color="#2C3E50",
                                lw=1.5, mutation_scale=14))
    mid_x = (x[0] + x[1]) / 2
    mid_y = (vals[0] + vals[1]) / 2
    ax.text(mid_x + 0.05, mid_y, f"{sign}{delta:.4f}",
            fontsize=9, color="#2C3E50", fontstyle="italic")

    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=11)
    ax.set_ylabel("F1 Score", fontsize=10)
    ax.set_title(f"F1 @ Mask {iou_tag}", fontsize=11, fontweight="bold")
    ax.set_ylim(ylim_lo, 1.005)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.suptitle("Generalisation: Validation vs Held-Out Test — FlowerAmodal v3",
             fontsize=10, y=1.02)
plt.tight_layout()
plt.savefig(OUT, dpi=300, bbox_inches="tight")
print(f"Saved {OUT}")
