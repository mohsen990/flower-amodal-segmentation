"""Generate F1 convergence plot across checkpoints for the paper."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

OUT = os.path.join(os.path.dirname(__file__), "fig_f1_convergence.pdf")

iters = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30]   # ×1000

f1_50 = [0.8922, 0.9229, 0.9453, 0.9495, 0.9497,
          0.9691, 0.9765, 0.9769, 0.9799, 0.9800]
f1_75 = [0.6803, 0.7828, 0.8466, 0.8796, 0.8782,
          0.9233, 0.9072, 0.9504, 0.9552, 0.9567]

# test-set final values (iter 30k model)
test_50 = 0.9834
test_75 = 0.9548

fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=False)

for ax, f1_vals, test_val, iou_tag, ylim_lo in zip(
        axes,
        [f1_50, f1_75],
        [test_50, test_75],
        ["IoU ≥ 0.50", "IoU ≥ 0.75"],
        [0.65, 0.60]):

    ax.plot(iters, f1_vals, "o-", color="#4C72B0", linewidth=2,
            markersize=6, label="Val F1", zorder=3)
    ax.axhline(test_val, color="#C44E52", linewidth=1.8,
               linestyle="--", label=f"Test F1 = {test_val:.4f}", zorder=2)

    # annotate last val point
    ax.annotate(f"{f1_vals[-1]:.4f}",
                xy=(iters[-1], f1_vals[-1]),
                xytext=(iters[-1] - 3, f1_vals[-1] - 0.025),
                fontsize=8, color="#4C72B0",
                arrowprops=dict(arrowstyle="->", color="#4C72B0", lw=0.8))

    ax.set_xlabel("Training Iterations (×1000)", fontsize=10)
    ax.set_ylabel("F1 Score", fontsize=10)
    ax.set_title(f"F1 @ Mask {iou_tag}", fontsize=11, fontweight="bold")
    ax.set_xticks(iters)
    ax.set_ylim(ylim_lo, 1.01)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8.5, loc="lower right")

fig.suptitle("Validation F1 Convergence — FlowerAmodal v3 (N\\_LAYERS=2, FP16, 30k iter)",
             fontsize=9.5, y=1.01)
plt.tight_layout()
plt.savefig(OUT, dpi=300, bbox_inches="tight")
print(f"Saved {OUT}")
