"""
plot_ablation_decay.py
Line plot showing how each method's correlation-at-K decays across cutoffs.
Visualises Table V (unsupervised SAE ablation, OOS).

Run:  python plot_ablation_decay.py
Out:  figures/ablation_cak_decay.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# ── Data from Table V / ablation_unsupervised_sae.py (OOS) ──────────────
cutoffs = [0.5, 1.0, 2.0, 5.0, 10.0]

methods = [
    ("WSFS-SAF",                "#4878A8", "o", 2.0,
     [0.4053, 0.3789, 0.3522, 0.3206, 0.2963]),
    ("SAE & Cosine Similarity", "#D48B6A", "s", 1.2,
     [0.4316, 0.3801, 0.3285, 0.2675, 0.2294]),
    ("SAE & Dot Product",       "#7A9A6E", "^", 1.2,
     [0.2005, 0.2039, 0.2028, 0.1962, 0.1861]),
    ("Molinari et al.",         "#8C8C8C", "D", 1.2,
     [0.1592, 0.1598, 0.1755, 0.1832, 0.1798]),
]

# ── Plot (matching norm_plot.py style) ───────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))

for name, color, marker, lw, vals in methods:
    ax.plot(cutoffs, vals, color=color, marker=marker, markersize=5,
            linewidth=lw, label=name)

ax.set_xlabel("Top-K%", fontsize=12)
ax.set_ylabel("Correlation-at-K", fontsize=12)
ax.set_title("Correlation-at-K Decay Across Methods (OOS)", fontsize=13)
ax.set_xticks(cutoffs)
ax.set_xticklabels([f"{c:.1f}%" for c in cutoffs])
ax.legend(fontsize=10, loc="upper right")

fig.tight_layout()

os.makedirs("figures", exist_ok=True)
fig.savefig("figures/ablation_cak_decay.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved -> figures/ablation_cak_decay.png")
