#!/usr/bin/env python3
"""
Top-N features by predictive score with their most highly-activating companies.

  !python interpret_features.py \
      --features-pkl "$path" --load-model "$path_sm" \
      --cov-ds /content/cov_dataset.parquet

  Add --discover-columns to inspect available columns first.
"""

import argparse, textwrap, gc
import numpy as np, pandas as pd, joblib


def unwrap_feature(x):
    while hasattr(x, '__len__') and len(x) == 1:
        x = x[0]
    if isinstance(x, np.ndarray):
        return x.astype(np.float32).flatten()
    return np.array(x, dtype=np.float32).flatten()


def find_col(df, candidates):
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


P = argparse.ArgumentParser()
P.add_argument("--features-pkl", required=True)
P.add_argument("--load-model", required=True)
P.add_argument("--cov-ds", default=None)
P.add_argument("--n-features", type=int, default=20)
P.add_argument("--n-companies", type=int, default=5)
P.add_argument("--desc-width", type=int, default=60)
P.add_argument("--desc-chars", type=int, default=300)
P.add_argument("--discover-columns", action="store_true")
args = P.parse_args()

# ── load ─────────────────────────────────────────────────────────────

df_f = pd.read_pickle(args.features_pkl)
df_f["features"] = df_f["features"].apply(unwrap_feature)
df_f["__index_level_0__"] = df_f["__index_level_0__"].astype(str)

scores = joblib.load(args.load_model)["scores"]

COV_HF = (
    "Mateusz1017/annual_reports_tokenized_llama3_logged_returns"
    "_no_null_returns_and_incomplete_descriptions_24k"
)
if args.cov_ds:
    df_c = pd.read_parquet(args.cov_ds)
else:
    from datasets import load_dataset as _lds
    df_c = _lds(COV_HF)["train"].to_pandas()

df_c["__index_level_0__"] = df_c["__index_level_0__"].astype(str)
df_c["year"] = df_c["year"].astype(int)

if args.discover_columns:
    print(f"\nCovariate columns ({len(df_c.columns)}):")
    for c in df_c.columns:
        vals = df_c[c].dropna()
        sample = repr(vals.iloc[0])[:80] if len(vals) > 0 else "N/A"
        print(f"  {c:40s}  dtype={str(df_c[c].dtype):10s}  sample={sample}")
    print(f"\nFeatures pkl columns: {list(df_f.columns)}")
    print(f"Scores shape: {scores.shape},  positive: {(scores > 0).sum()}")
    exit(0)

# ── detect columns ───────────────────────────────────────────────────

name_col = find_col(df_c, ["company_name", "companyname", "name",
                            "comp_name", "ticker"])
desc_col = find_col(df_c, ["description", "text", "filing_text",
                            "business_description"])
sic_col  = find_col(df_c, ["sic_code", "sic", "siccode"])

if desc_col is None:
    try:
        from datasets import load_dataset as _lds
        df_full = _lds("marco-molinari/company_reports_with_features"
                        )["train"].to_pandas()
        df_full["__index_level_0__"] = df_full["__index_level_0__"].astype(str)

        desc_col = find_col(df_full, ["description", "text", "filing_text",
                                       "business_description"])
        if desc_col is None:
            for c in df_full.columns:
                if c in ("features", "__index_level_0__"):
                    continue
                if df_full[c].dtype == object:
                    vals = df_full[c].dropna()
                    if len(vals) > 0 and len(str(vals.iloc[0])) > 200:
                        desc_col = c
                        break

        if name_col is None:
            name_col = find_col(df_full, ["company_name", "companyname", "name"])

        merge_cols = ["__index_level_0__"]
        if desc_col and desc_col in df_full.columns:
            merge_cols.append(desc_col)
        if name_col and name_col in df_full.columns and name_col not in df_c.columns:
            merge_cols.append(name_col)
        merge_cols = list(dict.fromkeys(merge_cols))

        if len(merge_cols) > 1:
            df_c = df_c.merge(
                df_full[merge_cols].drop_duplicates(subset=["__index_level_0__"]),
                on="__index_level_0__", how="left")

        del df_full; gc.collect()
    except Exception:
        pass

# ── merge ────────────────────────────────────────────────────────────

df = pd.merge(df_f, df_c, on="__index_level_0__", how="inner")
df["year"] = df["year"].astype(int)
del df_f, df_c; gc.collect()

feat_matrix = np.vstack(df["features"].values)
nan_mask = np.isnan(feat_matrix).any(axis=1) | np.isinf(feat_matrix).any(axis=1)
if nan_mask.sum() > 0:
    df = df[~nan_mask].reset_index(drop=True)
    feat_matrix = feat_matrix[~nan_mask]

# ── top features ─────────────────────────────────────────────────────

ranked = np.argsort(scores)[::-1]
top_feats = [int(i) for i in ranked if scores[i] > 0][:args.n_features]

# ── helpers ──────────────────────────────────────────────────────────

def get_label(row):
    if name_col and name_col in row.index and pd.notna(row[name_col]):
        return str(row[name_col]).strip()
    return str(row["__index_level_0__"])

def get_desc(row):
    if desc_col and desc_col in row.index and pd.notna(row[desc_col]):
        d = " ".join(str(row[desc_col]).split())
        if args.desc_chars > 0 and len(d) > args.desc_chars:
            d = d[:args.desc_chars] + "…"
        return d
    return ""

def get_sic(row):
    if sic_col and sic_col in row.index and pd.notna(row[sic_col]):
        try: return str(int(row[sic_col]))
        except (ValueError, TypeError): return str(row[sic_col])
    return ""

# ── table ────────────────────────────────────────────────────────────

has_desc = desc_col is not None and desc_col in df.columns
has_sic  = sic_col is not None and sic_col in df.columns

W = {"feat": 24, "name": 30, "year": 6, "sic": 6, "act": 10}
if has_desc: W["desc"] = args.desc_width

col_order = ["feat", "name", "year"]
if has_sic: col_order.append("sic")
col_order.append("act")
if has_desc: col_order.append("desc")

col_headers = {"feat": "Feature", "name": "Company", "year": "Year",
               "sic": "SIC", "act": "Activ.", "desc": "Description"}

widths = [W[c] for c in col_order]
hline = lambda l, m, r: l + m.join("─" * w for w in widths) + r
cell  = lambda t, w: f"{str(t)[:w]:<{w}}"

print()
print(hline("┌", "┬", "┐"))
print("│" + "│".join(cell(f" {col_headers[c]}", W[c]) for c in col_order) + "│")
print(hline("├", "┼", "┤"))

for rank, feat_idx in enumerate(top_feats):
    col_data = feat_matrix[:, feat_idx]
    active_idx = np.where(col_data > 0)[0]
    n_active = len(active_idx)
    if n_active == 0:
        continue

    sorted_active = active_idx[np.argsort(col_data[active_idx])[::-1]]

    seen = set()
    companies = []
    for ci in sorted_active:
        row = df.iloc[ci]
        lbl = get_label(row)
        if lbl in seen:
            continue
        seen.add(lbl)
        companies.append({"name": lbl, "year": int(row["year"]),
                          "sic": get_sic(row), "act": col_data[ci],
                          "desc": get_desc(row)})
        if len(companies) >= args.n_companies:
            break

    if rank > 0:
        print(hline("├", "┼", "┤"))

    for ci, comp in enumerate(companies):
        feat_lines = ([f" #{rank+1}: idx {feat_idx}",
                       f" s={scores[feat_idx]:.4f}",
                       f" n={n_active}"] if ci == 0 else [])

        desc_lines = (textwrap.wrap(comp["desc"], width=W["desc"] - 2) or [""]
                      ) if has_desc and comp["desc"] else [""]

        n_lines = max(len(desc_lines), len(feat_lines) if ci == 0 else 1)

        for ln in range(n_lines):
            parts = []
            for c in col_order:
                w = W[c]
                if c == "feat":
                    t = feat_lines[ln] if ci == 0 and ln < len(feat_lines) else ""
                elif c == "name":
                    t = f" {comp['name']}" if ln == 0 else ""
                elif c == "year":
                    t = f" {comp['year']}" if ln == 0 else ""
                elif c == "sic":
                    t = f" {comp['sic']}" if ln == 0 else ""
                elif c == "act":
                    t = f" {comp['act']:.1f}" if ln == 0 else ""
                elif c == "desc":
                    d = desc_lines[ln] if ln < len(desc_lines) else ""
                    t = f" {d}"
                else:
                    t = ""
                parts.append(cell(t, w))
            print("│" + "│".join(parts) + "│")

print(hline("└", "┴", "┘"))