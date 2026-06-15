"""TP / FP / FN progression across checkpoints at IoU=0.5 and IoU=0.75."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

OUT = os.path.join(os.path.dirname(__file__), "fig_tp_fp_fn.pdf")

iters = [3, 6, 9, 12, 15, 18, 21, 24, 27, 30]

tp50 = [2690, 2837, 2826, 2887, 2858, 2919, 2973, 2945, 2943, 2959]
fp50 = [308,  279,  121,  162,  129,   73,   84,   52,   32,   48]
fn50 = [342,  195,  206,  145,  174,  113,   59,   87,   89,   73]

tp75 = [2051, 2354, 2531, 2645, 2595, 2733, 2722, 2865, 2869, 2871]
fp75 = [947,  628,  416,  337,  283,  155,  247,  132,  106,   99]
fn75 = [981,  678,  501,  387,  437,  299,  310,  167,  163,  161]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

for ax, (tp, fp, fn), iou_tag in zip(
        axes,
        [(tp50, fp50, fn50), (tp75, fp75, fn75)],
        ["IoU ≥ 0.50", "IoU ≥ 0.75"]):

    ax.plot(iters, tp, "o-",  color="#2E86AB", linewidth=2, markersize=5,
            label="TP", zorder=3)
    ax.plot(iters, fp, "s--", color="#E84855", linewidth=1.8, markersize=5,
            label="FP", zorder=3)
    ax.plot(iters, fn, "^--", color="#F18F01", linewidth=1.8, markersize=5,
            label="FN", zorder=3)

    # annotate final values
    for vals, color, offset in [(tp, "#2E86AB", 30), (fp, "#E84855", -60), (fn, "#F18F01", 30)]:
        ax.annotate(str(vals[-1]),
                    xy=(iters[-1], vals[-1]),
                    xytext=(iters[-1] - 1.5, vals[-1] + offset),
                    fontsize=8, color=color,
                    arrowprops=dict(arrowstyle="->", color=color, lw=0.7))

    ax.set_xlabel("Training Iterations (×1000)", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title(f"TP / FP / FN @ Mask {iou_tag}", fontsize=11, fontweight="bold")
    ax.set_xticks(iters)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9, loc="center right")

fig.suptitle("Detection Quality Across Checkpoints — FlowerAmodal v3 (val, 3,032 GT)",
             fontsize=10, y=1.01)
plt.tight_layout()
plt.savefig(OUT, dpi=300, bbox_inches="tight")
print(f"Saved {OUT}")
