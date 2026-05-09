"""
plot_per_year.py
Grouped bar chart of per-year Top-1% C@K vs. population mean correlation.
Supports the 2020 discussion in Section IV-C-3.

Run:  python plot_per_year.py
Out:  figures/per_year_performance.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# ── Data from Table VI / ablation_robustness.py ──────────────────────────
years    = [2014,   2015,   2016,   2017,   2018,   2019,   2020]
top1_cak = [0.2248, 0.2733, 0.1770, 0.0381, 0.3737, 0.2835, 0.5975]
pop_mean = [0.1272, 0.1142, 0.0949, 0.0492, 0.2128, 0.1265, 0.3354]

# ── Plot (matching norm_plot.py style) ───────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))

x = np.arange(len(years))
w = 0.32

ax.bar(x - w/2, top1_cak, w, color="#4878A8",
       label="WSFS-SAF top-1% C@K")
ax.bar(x + w/2, pop_mean, w, color="#C0C0C0",
       label="Population mean correlation")

ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Mean Return Correlation", fontsize=12)
ax.set_title("Per-Year Top-1% Correlation-at-K (OOS)", fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(years)
ax.legend(fontsize=10, loc="upper left")

fig.tight_layout()

os.makedirs("figures", exist_ok=True)
fig.savefig("figures/per_year_performance.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("Saved -> figures/per_year_performance.png")
