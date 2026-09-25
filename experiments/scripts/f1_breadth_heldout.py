#!/usr/bin/env python3
"""Figure 1 (PVLDB): breadth fails by generalization, not by budget.

Left  : % of HELD-OUT queries whose entire evidence set is already resident when
        the hot set is fitted on the other half of the workload with NO budget cap
        (it materializes every object any training query touched).
Right : % of a held-out query's evidence objects that no training query ever
        touched -- the mechanism behind the left panel.
Numbers: analyze_breadth_heldout.py, 20 random 50/50 splits.
Depth storage shares (b=8, every object) from the exposure recount.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ORDER = ["Agriculture", "Medical", "Novel", "Mix", "Legal"]
SERV   = {"Agriculture": 0.4, "Medical": 19.0, "Novel": 2.3, "Mix": 4.5, "Legal": 0.0}
UNSEEN = {"Agriculture": 84.1, "Medical": 47.9, "Novel": 80.2, "Mix": 71.7, "Legal": 84.2}
STOR   = {"Agriculture": 14.9, "Medical": 28.2, "Novel": 29.1, "Mix": 40.7, "Legal": 5.4}
COL = {"Agriculture": "#1b6ca8", "Medical": "#12897b", "Novel": "#8a5fbf",
       "Legal": "#b5651d", "Mix": "#c0392b"}

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 8.5, "axes.labelsize": 8.5,
    "xtick.labelsize": 7.6, "ytick.labelsize": 7.8, "axes.linewidth": 0.8,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})
fig, (axL, axR) = plt.subplots(1, 2, figsize=(3.4, 2.15))
x = np.arange(len(ORDER))
cols = [COL[n] for n in ORDER]

# ---- left: held-out servability -------------------------------------------
axL.axhline(100, color="0.25", lw=0.9, ls=(0, (4, 2.5)), zorder=2)
axL.bar(x, [SERV[n] for n in ORDER], width=0.62, color=cols, zorder=3)
for i, n in enumerate(ORDER):
    axL.text(i, SERV[n] + 3.5, f"{SERV[n]:.1f}", ha="center", va="bottom",
             fontsize=6.8, color="0.15")
axL.text((len(ORDER) - 1) / 2, 103, "depth ($b{=}8$): 100%",
         ha="center", va="bottom", fontsize=6.9, color="0.15")
axL.set_ylabel("held-out queries\nservable offline (%)")
axL.set_ylim(0, 128); axL.set_yticks([0, 50, 100])
axL.set_title("breadth does not generalize", fontsize=7.8, pad=3)

# ---- right: why ------------------------------------------------------------
axR.bar(x, [UNSEEN[n] for n in ORDER], width=0.62, color=cols, alpha=0.75, zorder=3)
for i, n in enumerate(ORDER):
    axR.text(i, UNSEEN[n] + 3.5, f"{UNSEEN[n]:.0f}", ha="center", va="bottom",
             fontsize=6.8, color="0.15")
axR.set_ylabel("held-out evidence\nnever seen before (%)")
axR.set_ylim(0, 128); axR.set_yticks([0, 50, 100])
axR.set_title("because the objects are new", fontsize=7.8, pad=3)

for ax in (axL, axR):
    ax.set_xticks(x)
    ax.set_xticklabels(["Agri", "Med", "Novel", "Mix", "Legal"],
                       rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.25, lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

fig.tight_layout(pad=0.2, w_pad=1.6)
out = "F1_breadth_depth.pdf"
fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
print("wrote", out, "| depth storage shares:", STOR)
