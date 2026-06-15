"""Generate dataset construction pipeline diagram."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import os

OUT = os.path.join(os.path.dirname(__file__), "fig_pipeline.pdf")

fig, ax = plt.subplots(figsize=(11, 3.8))
ax.set_xlim(0, 11)
ax.set_ylim(0, 4)
ax.axis("off")

# ── colour palette ─────────────────────────────────────────────────────────
C_BOX    = "#EBF4FA"
C_BORDER = "#2E86C1"
C_ARROW  = "#2C3E50"
C_GREEN  = "#D5F5E3"
C_GBORD  = "#1E8449"
C_ORANGE = "#FDEBD0"
C_OBORD  = "#CA6F1E"
C_PURPLE = "#EAE3F7"
C_PBORD  = "#6C3483"

def box(ax, x, y, w, h, label, sublabel="", fc=C_BOX, ec=C_BORDER, fs=9):
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                          fc=fc, ec=ec, lw=1.5, zorder=2)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2 + (0.12 if sublabel else 0),
            label, ha="center", va="center", fontsize=fs,
            fontweight="bold", zorder=3)
    if sublabel:
        ax.text(x + w/2, y + h/2 - 0.22, sublabel,
                ha="center", va="center", fontsize=7.2,
                color="#555555", zorder=3)

def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=C_ARROW,
                                lw=1.8, mutation_scale=14), zorder=4)

# ── title ──────────────────────────────────────────────────────────────────
ax.text(5.5, 3.75, "FlowerAmodal v3 — CopyPaste Dataset Construction Pipeline",
        ha="center", va="center", fontsize=11, fontweight="bold")

# ── step 1: source images ──────────────────────────────────────────────────
box(ax, 0.15, 1.6, 1.5, 1.1, "Source Images",
    "Oxford 102\n2 categories", fc="#FEF9E7", ec="#B7950B")
ax.text(0.90, 1.3, "6,062 train\n1,516 val\n1,586 test",
        ha="center", va="center", fontsize=7, color="#555555")

arrow(ax, 1.65, 2.15, 2.25, 2.15)

# ── step 2: select pair ────────────────────────────────────────────────────
box(ax, 2.25, 1.6, 1.5, 1.1, "Select Pair",
    "Flower A (bg)\nFlower B (fg)", fc=C_BOX, ec=C_BORDER)

arrow(ax, 3.75, 2.15, 4.35, 2.15)

# ── step 3: shift & overlap ────────────────────────────────────────────────
box(ax, 4.35, 1.6, 1.6, 1.1, "Random Shift",
    "Overlap 10–80%\nof A's area", fc=C_BOX, ec=C_BORDER)

arrow(ax, 5.95, 2.15, 6.55, 2.15)

# ── step 4: composite ─────────────────────────────────────────────────────
box(ax, 6.55, 1.6, 1.5, 1.1, "Composite",
    "B pasted over A\nReal background", fc=C_GREEN, ec=C_GBORD)

arrow(ax, 8.05, 2.15, 8.65, 2.15)

# ── step 5: annotations ───────────────────────────────────────────────────
box(ax, 8.65, 2.55, 2.2, 0.55, "Flower A (occluded)",
    "amodal / visible / occluder masks", fc=C_ORANGE, ec=C_OBORD, fs=8)
box(ax, 8.65, 1.85, 2.2, 0.55, "Flower B (occluder)",
    "amodal = visible, occluder = ∅", fc=C_PURPLE, ec=C_PBORD, fs=8)
box(ax, 8.65, 1.15, 2.2, 0.55, "is_occluder flag",
    "B → loss weight 0.1×", fc="#FDEDEC", ec="#922B21", fs=8)

# arrows to annotation boxes
arrow(ax, 8.65, 2.35, 8.65, 2.90)   # to Flower A box (adjusted)
arrow(ax, 8.65, 2.15, 8.65, 2.15)   # middle (already drawn by composite→)
# re-do the final arrow split
ax.annotate("", xy=(8.65, 2.90), xytext=(8.30, 2.90),
            arrowprops=dict(arrowstyle="-|>", color=C_ARROW, lw=1.4,
                            mutation_scale=12), zorder=4)
ax.annotate("", xy=(8.65, 2.12), xytext=(8.30, 2.12),
            arrowprops=dict(arrowstyle="-|>", color=C_ARROW, lw=1.4,
                            mutation_scale=12), zorder=4)
ax.annotate("", xy=(8.65, 1.42), xytext=(8.30, 1.42),
            arrowprops=dict(arrowstyle="-|>", color=C_ARROW, lw=1.4,
                            mutation_scale=12), zorder=4)
# vertical line from composite exit
ax.plot([8.30, 8.30], [1.42, 2.90], color=C_ARROW, lw=1.4, zorder=3)
ax.plot([8.05, 8.30], [2.15, 2.15], color=C_ARROW, lw=1.4, zorder=3)

# ── overlap range label ────────────────────────────────────────────────────
ax.text(5.15, 1.45, "Reject if outside\n10–80% range",
        ha="center", va="center", fontsize=7, color="#7D6608",
        style="italic")

plt.tight_layout()
plt.savefig(OUT, dpi=300, bbox_inches="tight")
print(f"Saved {OUT}")
