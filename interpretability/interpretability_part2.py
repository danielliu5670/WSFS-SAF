#!/usr/bin/env python3

import argparse
import numpy as np
import pandas as pd
import joblib
from scipy.stats import chi2_contingency
from tabulate import tabulate, SEPARATING_LINE
import gc


def unwrap_feature(x):
    while hasattr(x, '__len__') and len(x) == 1:
        x = x[0]
    if isinstance(x, np.ndarray):
        return x.astype(np.float32).flatten()
    return np.array(x, dtype=np.float32).flatten()


def bh_adjust(pvals):
    m = len(pvals)
    if m == 0:
        return np.array([])
    order = np.argsort(pvals)
    sorted_p = pvals[order]
    adj = np.zeros(m)
    adj[m - 1] = min(sorted_p[m - 1], 1.0)
    for i in range(m - 2, -1, -1):
        raw = sorted_p[i] * m / (i + 1)
        adj[i] = min(raw, adj[i + 1], 1.0)
    result = np.zeros(m)
    for i in range(m):
        result[order[i]] = adj[i]
    return result


def _fix_separators(table_str):
    lines = table_str.split('\n')
    sep = None
    for line in lines:
        if line.lstrip().startswith('├') and '┼' in line:
            sep = line
            break
    if sep is None:
        return table_str
    fixed = []
    for line in lines:
        stripped = line.replace('│', '').replace(' ', '').replace('\x01', '')
        if (stripped == '' and '│' in line
                and not line.lstrip().startswith('┌')
                and not line.lstrip().startswith('└')
                and not line.lstrip().startswith('├')):
            fixed.append(sep)
        else:
            fixed.append(line)
    return '\n'.join(fixed)


P = argparse.ArgumentParser()
P.add_argument("--features-pkl", required=True)
P.add_argument("--load-model", required=True)
P.add_argument("--n-features", type=int, default=20)
P.add_argument("--top-sics", type=int, default=3)
P.add_argument("--alpha", type=float, default=0.05)
P.add_argument(
    "--cov-ds",
    default=(
        "Mateusz1017/annual_reports_tokenized_llama3_logged_returns"
        "_no_null_returns_and_incomplete_descriptions_24k"
    ),
)
args = P.parse_args()

df_f = pd.read_pickle(args.features_pkl)
df_f["features"] = df_f["features"].apply(unwrap_feature)
df_f["__index_level_0__"] = df_f["__index_level_0__"].astype(str)

df_c = pd.read_parquet(args.cov_ds)
df_c["__index_level_0__"] = df_c["__index_level_0__"].astype(str)
df_c["year"] = df_c["year"].astype(int)

df = pd.merge(df_f, df_c, on="__index_level_0__", how="inner")
df = df.dropna(subset=["sic_code"])
df["year"] = df["year"].astype(int)
del df_f, df_c
gc.collect()

feat_matrix = np.vstack(df["features"].values)
nan_mask = np.isnan(feat_matrix).any(axis=1) | np.isinf(feat_matrix).any(axis=1)
if nan_mask.sum() > 0:
    df = df[~nan_mask].reset_index(drop=True)
    feat_matrix = feat_matrix[~nan_mask]

scores = joblib.load(args.load_model)["scores"]
ranked = np.argsort(scores)[::-1]
top_feats = [int(i) for i in ranked if scores[i] > 0][:args.n_features]

n_total = len(df)

sic_arr = []
for s in df["sic_code"].values:
    try:
        sic_arr.append(str(int(s)))
    except (ValueError, TypeError):
        sic_arr.append("")
sic_arr = np.array(sic_arr)

unique_sics = sorted(set(s for s in sic_arr if s != ""))

sic_masks = {}
for s in unique_sics:
    sic_masks[s] = (sic_arr == s)

results = []
for feat_idx in top_feats:
    col = feat_matrix[:, feat_idx]
    active = col > 0
    n_active = int(active.sum())
    if n_active == 0:
        continue

    for s in unique_sics:
        in_sic = sic_masks[s]
        n_sic = int(in_sic.sum())
        if n_sic < 5:
            continue

        a = int((active & in_sic).sum())
        b = int((active & ~in_sic).sum())
        c = int((~active & in_sic).sum())
        d = int((~active & ~in_sic).sum())

        exp_a = n_active * n_sic / n_total
        if exp_a < 1.0 or a <= exp_a:
            continue

        table = np.array([[a, b], [c, d]])
        try:
            chi2, pval, _, _ = chi2_contingency(table)
        except Exception:
            continue

        results.append({
            "feat": feat_idx,
            "score": scores[feat_idx],
            "sic": s,
            "n_active": n_active,
            "obs": a,
            "exp": exp_a,
            "ratio": a / exp_a,
            "pval": pval,
        })

if len(results) == 0:
    print("No enrichments found.")
    exit(0)

pvals = np.array([r["pval"] for r in results])
adj = bh_adjust(pvals)
for i in range(len(results)):
    results[i]["p_adj"] = adj[i]

rows = []
for feat_idx in top_feats:
    feat_res = [r for r in results if r["feat"] == feat_idx]
    ratios = [r["ratio"] for r in feat_res]
    order = np.argsort(ratios)[::-1]
    feat_res = [feat_res[int(i)] for i in order]
    top = feat_res[:args.top_sics]

    if len(top) == 0:
        rows.append([
            str(feat_idx),
            f"{scores[feat_idx]:.4f}",
            "", "", "", "", "", "", "",
        ])
        rows.append(SEPARATING_LINE)
        continue

    for i in range(len(top)):
        r = top[i]
        if r["p_adj"] < 0.001:
            sig = "***"
            p_str = "<0.001"
        elif r["p_adj"] < 0.01:
            sig = "**"
            p_str = f"{r['p_adj']:.3f}"
        elif r["p_adj"] < args.alpha:
            sig = "*"
            p_str = f"{r['p_adj']:.3f}"
        else:
            sig = ""
            p_str = f"{r['p_adj']:.3f}"

        feat_str = str(feat_idx) if i == 0 else ""
        score_str = f"{r['score']:.4f}" if i == 0 else ""
        active_str = str(r["n_active"]) if i == 0 else ""

        rows.append([
            feat_str, score_str, active_str,
            r["sic"], str(r["obs"]), f"{r['exp']:.1f}",
            f"{r['ratio']:.1f}x", p_str, sig,
        ])

    rows.append(SEPARATING_LINE)

if rows and rows[-1] is SEPARATING_LINE:
    rows.pop()

headers = ["Feature", "Score", "Active", "SIC", "Obs", "Exp",
           "Ratio", "p_adj", ""]

n_sig = 0
for r in results:
    if r["p_adj"] < args.alpha:
        n_sig += 1

print()
print(_fix_separators(tabulate(
    rows, headers=headers, tablefmt="simple_outline",
)))
print(f"\n{n_sig} / {len(results)} enrichments significant "
      f"(BH-adjusted, α={args.alpha})")