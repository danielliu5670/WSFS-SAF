"""
plot_alpha_sensitivity.py
Line plot showing Top-1% C@K as a function of alpha, one line per K value.
K=1000 (the selected configuration) is emphasized; others shown as context.

Run:  python plot_alpha_sensitivity.py
Out:  figures/alpha_sensitivity.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# ── Data from the full parameter grid (project summary, all-years) ───────
alphas = [0.0, 0.25, 0.5, 0.75, 1.0]

# top-1% C@K for each (K, alpha) pair
grid = [
    ( 500, [0.3331, 0.3381, 0.3404, 0.3325, 0.2949]),
    ( 750, [0.3271, 0.3316, 0.3316, 0.3233, 0.2912]),
    (1000, [0.3763, 0.3745, 0.3715, 0.3636, 0.3016]),
    (1250, [0.3762, 0.3743, 0.3710, 0.3621, 0.3044]),
    (1500, [0.3783, 0.3756, 0.3702, 0.3577, 0.2951]),
    (2000, [0.3789, 0.3751, 0.3674, 0.3529, 0.2961]),
    (2500, [0.3766, 0.3742, 0.3675, 0.3526, 0.3002]),
]

# ── Plot (matching norm_plot.py style) ───────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))

# Background lines first (other K values)
for k, vals in grid:
    if k != 2000:
        ax.plot(alphas, vals, color="#C0C0C0", marker="o", markersize=4,
                linewidth=1.0, label=f"K = {k}")

# Emphasized line on top
for k, vals in grid:
    if k == 2000:
        ax.plot(alphas, vals, color="#4878A8", marker="o", markersize=5,
                linewidth=2.0, label=f"K = {k}", zorder=3)

ax.set_xlabel(r"$\alpha$", fontsize=12)
ax.set_ylabel("Top-1% Correlation-at-K", fontsize=12)
ax.set_title(r"Sensitivity of Top-1% C@K to $\alpha$ and K", fontsize=13)
ax.set_xticks(alphas)
ax.legend(fontsize=10, loc="lower left")

fig.tight_layout()

os.makedirs("figures", exist_ok=True)
fig.savefig("figures/alpha_sensitivity.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved -> figures/alpha_sensitivity.png")
