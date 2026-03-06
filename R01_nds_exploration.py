"""
R01 NDS Exploration — Spyder-friendly cell-mode script.

Pipeline:
  1. Load data (Excel)
  2. Preprocess NDS (extract max scores, merge 24morning into 24hr)
  3. Plot NDS distributions:
       i.   4hr NDS overall
       ii.  24hr NDS overall
       iii. 4hr NDS by asphyxia group
       iv.  24hr NDS by asphyxia group
  4. Feature extraction & nested-CV classification
       a. Build feature table (BP normalisation)
       b. Create binary labels (within-group percentile)
       c. Compare Elastic-Net Logistic, Random Forest, XGBoost
          via correlation pruning + embedded selection inside nested CV
          (heavy lifting lives in ml_pipeline.py)
"""

#%% ============================================================
# IMPORTS
# ==============================================================
import os
import sys
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

#%% ============================================================
# CONFIGURATION
# ==============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

EXCEL_PATH = os.path.join(DATA_DIR, "MASTER_SPREADSHEET.xlsx")
SHEET_NAMES = ["Filtered_Subset_No_REBOA", "No_REBOA_selected_column", "cleaned_no_reboa"]

GROUP_COL = "Intended Asphyxia/Asystole time (min)"

#%% ============================================================
# BLOCK 1 — LOAD DATA
# ==============================================================
print("=" * 60)
print("BLOCK 1: Load data")
print("=" * 60)

if not os.path.isfile(EXCEL_PATH):
    raise FileNotFoundError(
        f"Excel file not found: {EXCEL_PATH}\n"
        f"Place data.xlsx in {DATA_DIR} with sheets: {SHEET_NAMES}"
    )

dfs = {}
for name in SHEET_NAMES:
    dfs[name] = pd.read_excel(EXCEL_PATH, sheet_name=name, engine="openpyxl")
    print(f"  Loaded sheet '{name}': {len(dfs[name])} rows, {dfs[name].shape[1]} cols")

df_subset = dfs["Filtered_Subset_No_REBOA"]
print(f"\n  df_subset shape: {df_subset.shape}")
print(f"  Columns: {list(df_subset.columns)}")

#%% ============================================================
# BLOCK 2 — PREPROCESS NDS
# ==============================================================
print("\n" + "=" * 60)
print("BLOCK 2: Preprocess NDS scores")
print("=" * 60)


def extract_max_score(nds_string):
    """Extract the maximum integer from strings like '32(40)' or '27'."""
    if isinstance(nds_string, str):
        numbers = re.findall(r"\d+", nds_string)
        if numbers:
            return max(map(int, numbers))
    return None


# --- 4hr NDS ---
if "4hr NDS" in df_subset.columns:
    max_4hr_nds = df_subset["4hr NDS"].apply(extract_max_score)
    print(f"  4hr NDS: {max_4hr_nds.notna().sum()} valid / {len(max_4hr_nds)} total")
else:
    max_4hr_nds = pd.Series(dtype=float)
    print("  WARNING: '4hr NDS' column not found")

# --- 24hr NDS ---
if "24hr NDS" in df_subset.columns:
    max_24hr_nds = df_subset["24hr NDS"].apply(extract_max_score)
    max_24hr_nds = max_24hr_nds.fillna(60)
    print(f"  24hr NDS: {max_24hr_nds.notna().sum()} valid / {len(max_24hr_nds)} total (NaN -> 60)")
else:
    max_24hr_nds = pd.Series(dtype=float)
    print("  WARNING: '24hr NDS' column not found")

# --- 24morning NDS: override 24hr where 24morning is higher ---
if "24morning NDS" in df_subset.columns and len(max_24hr_nds) > 0:
    max_24m_nds = df_subset["24morning NDS"].apply(extract_max_score)
    max_24hr_nds = max_24hr_nds.reindex(max_24m_nds.index)
    mask = max_24m_nds > max_24hr_nds
    max_24hr_nds = max_24hr_nds.where(~mask, max_24m_nds)
    print(f"  24morning override applied: {mask.sum()} values replaced")

# --- Asphyxia groups ---
if GROUP_COL in df_subset.columns:
    group_data = df_subset[GROUP_COL]
    if isinstance(group_data, pd.DataFrame):
        group_data = group_data.iloc[:, 0]
    groups = group_data.astype(str)
    unique_groups = sorted(groups.unique())
    print(f"  Asphyxia groups ({len(unique_groups)}): {unique_groups}")
else:
    groups = pd.Series(dtype=str)
    unique_groups = []
    print(f"  WARNING: '{GROUP_COL}' column not found")

#%% ============================================================
# BLOCK 3 — PLOTTING HELPERS
# ==============================================================


def _stat_legend(values):
    """Return a formatted string with median, mean, IQR, and n."""
    med = np.median(values)
    mean = np.mean(values)
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    n = len(values)
    return (
        f"Median: {med:.1f}\n"
        f"Mean:   {mean:.1f}\n"
        f"IQR:    {iqr:.1f} [{q1:.1f}–{q3:.1f}]\n"
        f"n = {n}"
    )


def _add_stat_overlay(ax, values):
    """Draw vertical lines for median, mean, Q1, Q3 on an axis."""
    med = np.median(values)
    mean = np.mean(values)
    q1, q3 = np.percentile(values, [25, 75])

    ax.axvline(med, color="crimson", linestyle="--", linewidth=1.5, label="Median")
    ax.axvline(mean, color="dodgerblue", linestyle="-.", linewidth=1.5, label="Mean")
    ax.axvspan(q1, q3, alpha=0.12, color="orange", label=f"IQR [{q1:.1f}–{q3:.1f}]")

    patch = mpatches.Patch(color="none", label=_stat_legend(values))
    handles, _ = ax.get_legend_handles_labels()
    handles.append(patch)
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)


#%% ============================================================
# BLOCK 3a — Plot i: 4hr NDS distribution
# ==============================================================
print("\n" + "=" * 60)
print("BLOCK 3a: 4hr NDS distribution")
print("=" * 60)

valid_4hr = max_4hr_nds.dropna().astype(float)

if len(valid_4hr) > 0:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(valid_4hr, bins=15, edgecolor="black", alpha=0.7, color="steelblue")
    ax.set_title("4hr NDS Distribution (max score)")
    ax.set_xlabel("NDS Score")
    ax.set_ylabel("Count")
    _add_stat_overlay(ax, valid_4hr.values)
    plt.tight_layout()
    plt.show()
else:
    print("  No valid 4hr NDS data to plot.")

#%% ============================================================
# BLOCK 3b — Plot ii: 24hr NDS distribution
# ==============================================================
print("\n" + "=" * 60)
print("BLOCK 3b: 24hr NDS distribution")
print("=" * 60)

valid_24hr = max_24hr_nds.dropna().astype(float)

if len(valid_24hr) > 0:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(valid_24hr, bins=15, edgecolor="black", alpha=0.7, color="darkorange")
    ax.set_title("24hr NDS Distribution (max score)")
    ax.set_xlabel("NDS Score")
    ax.set_ylabel("Count")
    _add_stat_overlay(ax, valid_24hr.values)
    plt.tight_layout()
    plt.show()
else:
    print("  No valid 24hr NDS data to plot.")

#%% ============================================================
# BLOCK 3c — Plot iii: 4hr NDS by asphyxia group
# ==============================================================
print("\n" + "=" * 60)
print("BLOCK 3c: 4hr NDS by asphyxia group")
print("=" * 60)

if len(valid_4hr) > 0 and len(unique_groups) > 0:
    n_groups = len(unique_groups)
    fig, axes = plt.subplots(1, n_groups, figsize=(5 * n_groups, 4), squeeze=False)
    axes = axes.ravel()

    for i, grp in enumerate(unique_groups):
        grp_mask = groups == grp
        grp_scores = max_4hr_nds[grp_mask].dropna().astype(float)

        if len(grp_scores) > 0:
            axes[i].hist(grp_scores, bins=10, edgecolor="black", alpha=0.7,
                         color="steelblue")
            _add_stat_overlay(axes[i], grp_scores.values)
        axes[i].set_title(f"4hr NDS — {GROUP_COL}: {grp}")
        axes[i].set_xlabel("NDS Score")
        axes[i].set_ylabel("Count")

    fig.suptitle("4hr NDS by Asphyxia Group", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.show()
else:
    print("  Insufficient data for grouped 4hr NDS plot.")

#%% ============================================================
# BLOCK 3d — Plot iv: 24hr NDS by asphyxia group
# ==============================================================
print("\n" + "=" * 60)
print("BLOCK 3d: 24hr NDS by asphyxia group")
print("=" * 60)

if len(valid_24hr) > 0 and len(unique_groups) > 0:
    n_groups = len(unique_groups)
    fig, axes = plt.subplots(1, n_groups, figsize=(5 * n_groups, 4), squeeze=False)
    axes = axes.ravel()

    for i, grp in enumerate(unique_groups):
        grp_mask = groups == grp
        grp_scores = max_24hr_nds[grp_mask].dropna().astype(float)

        if len(grp_scores) > 0:
            axes[i].hist(grp_scores, bins=10, edgecolor="black", alpha=0.7,
                         color="darkorange")
            _add_stat_overlay(axes[i], grp_scores.values)
        axes[i].set_title(f"24hr NDS — {GROUP_COL}: {grp}")
        axes[i].set_xlabel("NDS Score")
        axes[i].set_ylabel("Count")

    fig.suptitle("24hr NDS by Asphyxia Group", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.show()
else:
    print("  Insufficient data for grouped 24hr NDS plot.")

#%% ============================================================
# BLOCK 4a — BUILD FEATURE TABLE  (BP normalisation)
# ==============================================================
print("\n" + "=" * 60)
print("BLOCK 4a: Build feature table")
print("=" * 60)

df_features = dfs["No_REBOA_selected_column"].copy()

BP_NORM_CONFIG = {
    "DBP_baseline": [
        "DBP start (Asphyxia/KCl)", "DBP EEGflat",
        "MaxDBP (postROSC)", "MinDBP (postROSC)",
        "DBP5minpostROSC", "DBP30minpostROSC",
        "DBP for MaxSBP", "DBP for MinSBP",
    ],
    "SBP_baseline": [
        "SBP start (Asphyxia/KCl)", "SBP EEGflat",
        "MaxSBP (postROSC)", "MinSBP (postROSC)",
        "SBP for Min DBP", "SBP for Max DBP",
        "SBP5minpostROSC", "SBP30minpostROSC",
    ],
    "MAP_baseline": [
        "MAP start (Asphyxia/KCl, whole integer)", "MAP EEG flat",
        "MAP EEG Flat Formula", "MAP for MaxSBP",
        "MAP at the time of MinSBP", "MAP at the time of MaxDBP",
        "MAP at the time of MinDBP",
    ],
}

for base_col, norm_cols in BP_NORM_CONFIG.items():
    if base_col not in df_features.columns:
        continue
    for c in norm_cols + [base_col]:
        if c in df_features.columns:
            df_features[c] = pd.to_numeric(df_features[c], errors="coerce")
    if df_features[base_col].isna().any():
        df_features[base_col] = df_features[base_col].fillna(
            df_features[base_col].mean()
        )
    for c in norm_cols:
        if c in df_features.columns:
            df_features[c] = df_features[c] / df_features[base_col]

rat_ids_feat = (
    df_features["Rat ID"].astype(str) if "Rat ID" in df_features.columns else None
)

DROP_COLS = ["Rat ID", "Time from ROSC that BSR Reaches 0.5"]
X_all = df_features.drop(
    columns=[c for c in DROP_COLS if c in df_features.columns]
)
X_all = X_all.apply(pd.to_numeric, errors="coerce").fillna(0)
print(f"  Feature matrix: {X_all.shape}")

#%% ============================================================
# BLOCK 4b — CREATE BINARY LABELS  (within-group percentile)
# ==============================================================
print("\n" + "=" * 60)
print("BLOCK 4b: Create binary labels (group-percentile method)")
print("=" * 60)


def classify_by_group_percentile(df, score_series, group_col, percentile=50):
    """Classify as good/bad by within-group percentile."""
    if len(df) != len(score_series):
        raise ValueError("df and score_series must have the same length.")
    grp = df[group_col]
    if isinstance(grp, pd.DataFrame):
        grp = grp.iloc[:, 0]
    tmp = pd.DataFrame({group_col: grp.values, "score": score_series.values})
    tmp["label"] = np.nan
    tmp["label"] = tmp["label"].astype(object)
    for g in tmp[group_col].unique():
        idx = tmp[group_col].isna() if pd.isna(g) else (tmp[group_col] == g)
        valid = tmp.loc[idx, "score"].dropna()
        if valid.empty:
            tmp.loc[idx, "label"] = "bad"
        else:
            thr = np.percentile(valid, percentile)
            tmp.loc[idx, "label"] = np.where(
                tmp.loc[idx, "score"] > thr, "good", "bad"
            )
    return tmp["label"].values


cls_4hr = classify_by_group_percentile(df_subset, max_4hr_nds, GROUP_COL)
cls_24hr = classify_by_group_percentile(df_subset, max_24hr_nds, GROUP_COL)

print(f"  4hr  labels — good: {np.sum(cls_4hr == 'good')}, "
      f"bad: {np.sum(cls_4hr == 'bad')}")
print(f"  24hr labels — good: {np.sum(cls_24hr == 'good')}, "
      f"bad: {np.sum(cls_24hr == 'bad')}")

# --- Align features (No_REBOA_selected_column) with labels (Filtered_Subset)
#     via Rat ID merge ---
rat_ids_label = (
    df_subset["Rat ID"].astype(str) if "Rat ID" in df_subset.columns else None
)

if rat_ids_feat is not None and rat_ids_label is not None:
    label_df = pd.DataFrame({
        "Rat ID": rat_ids_label,
        "y_4hr":  (cls_4hr == "good").astype(int),
        "y_24hr": (cls_24hr == "good").astype(int),
    })
    merged = X_all.copy()
    merged["Rat ID"] = rat_ids_feat.values
    merged = merged.merge(label_df, on="Rat ID", how="inner")
    y_4hr  = merged["y_4hr"].values
    y_24hr = merged["y_24hr"].values
    X = merged.drop(columns=["Rat ID", "y_4hr", "y_24hr"])
else:
    n = min(len(X_all), len(cls_4hr))
    y_4hr  = (cls_4hr[:n] == "good").astype(int)
    y_24hr = (cls_24hr[:n] == "good").astype(int)
    X = X_all.iloc[:n]

print(f"  Aligned X: {X.shape},  y_4hr: {y_4hr.shape},  y_24hr: {y_24hr.shape}")

#%% ============================================================
# BLOCK 4c — NESTED CV MODEL COMPARISON
# ==============================================================
print("\n" + "=" * 60)
print("BLOCK 4c: Nested-CV model comparison")
print("=" * 60)

# Ensure the script directory is on sys.path so that ml_pipeline.py
# can be imported reliably when running cells in Spyder/IPython.
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from ml_pipeline import (  # noqa: E402
    run_repeated_nested_cv,
    print_repeated_cv_summary,
    plot_comparison,
)

feature_names = X.columns.tolist()

for target_name, y_target in [("4hr NDS", y_4hr), ("24hr NDS", y_24hr)]:
    n_pos = int(y_target.sum())
    n_neg = int(len(y_target) - n_pos)
    print(f"\n  --- Target: {target_name}  (pos={n_pos}, neg={n_neg}) ---")

    if n_pos < 2 or n_neg < 2:
        print(f"  SKIPPED — need ≥2 samples per class (got {n_pos}/{n_neg})")
        continue

    repeated = run_repeated_nested_cv(
        X,
        y_target,
        seeds=(11, 22, 33, 44, 55),
        outer_k=5,
        inner_k=5,
        corr_threshold=0.85,
        feature_names=feature_names,
        verbose=False,
    )
    print_repeated_cv_summary(repeated, target_name=target_name, min_freq=0.80)

    # Plot uses same scores structure
    plot_scores = {name: {"scores": repeated["raw_scores"][name]} for name in repeated["raw_scores"]}
    plot_comparison(plot_scores, target_name=target_name)

#%% ============================================================
# BLOCK 4d — NESTED CV WITHOUT FEATURE SELECTION (all features)
# ============================================================
print("\n" + "=" * 60)
print("BLOCK 4d: Nested-CV model comparison (no feature selection)")
print("=" * 60)

for target_name, y_target in [("4hr NDS", y_4hr), ("24hr NDS", y_24hr)]:
    n_pos = int(y_target.sum())
    n_neg = int(len(y_target) - n_pos)
    print(f"\n  --- Target: {target_name}  (pos={n_pos}, neg={n_neg}) ---")

    if n_pos < 2 or n_neg < 2:
        print(f"  SKIPPED — need ≥2 samples per class (got {n_pos}/{n_neg})")
        continue

    repeated_no_fs = run_repeated_nested_cv(
        X,
        y_target,
        seeds=(11, 22, 33, 44, 55),
        outer_k=5,
        inner_k=5,
        corr_threshold=0.85,
        feature_names=feature_names,
        verbose=False,
        feature_selection=False,
    )
    print_repeated_cv_summary(repeated_no_fs, target_name=target_name, min_freq=0.80)

    plot_scores_no_fs = {name: {"scores": repeated_no_fs["raw_scores"][name]} for name in repeated_no_fs["raw_scores"]}
    plot_comparison(plot_scores_no_fs, target_name=f"{target_name} (no FS)")

#%% ============================================================
# BLOCK 4e — FEATURE IMPORTANCE (Random Forest: MDI + permutation)
# ============================================================
print("\n" + "=" * 60)
print("BLOCK 4e: Feature importance (Random Forest)")
print("=" * 60)

from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

# Scale features for consistent RF (optional but often done)
from sklearn.preprocessing import StandardScaler  # noqa: E402

for target_name, y_target in [("4hr NDS", y_4hr), ("24hr NDS", y_24hr)]:
    if np.unique(y_target).size < 2:
        continue
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y_target, stratify=y_target, random_state=42, test_size=0.2
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    forest = RandomForestClassifier(n_estimators=100, random_state=42)
    forest.fit(X_train_s, y_train)

    # MDI: mean decrease in impurity (with std across trees)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    order = np.argsort(importances)[::-1]
    names_ordered = [feature_names[i] for i in order]
    imp_ordered = importances[order]
    std_ordered = std[order]

    # Permutation importance on test set
    perm = permutation_importance(
        forest, X_test_s, y_test, n_repeats=10, random_state=42, n_jobs=-1
    )
    perm_imp = perm.importances_mean[order]
    perm_std = perm.importances_std[order]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    n_show = min(20, len(feature_names))

    ax = axes[0]
    ax.barh(range(n_show), imp_ordered[:n_show], xerr=std_ordered[:n_show], align="center")
    ax.set_yticks(range(n_show))
    ax.set_yticklabels(names_ordered[:n_show], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Mean decrease in impurity")
    ax.set_title(f"{target_name} — Feature importances (MDI)")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.barh(range(n_show), perm_imp[:n_show], xerr=perm_std[:n_show], align="center", color="C1")
    ax.set_yticks(range(n_show))
    ax.set_yticklabels(names_ordered[:n_show], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Mean accuracy decrease")
    ax.set_title(f"{target_name} — Permutation importance")
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"Random Forest feature importance — {target_name}", fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.show()

#%% ============================================================
# BLOCK 5 — CORRELATION: 4hr NDS vs MAP at MinSBP and SBP5minpostROSC
# ==============================================================
print("\n" + "=" * 60)
print("BLOCK 5: Correlation of 4hr NDS with MAP at MinSBP and SBP5minpostROSC")
print("=" * 60)

# Build merged table: 4hr NDS (from df_subset) + raw BP cols from original sheet
df_raw = dfs["No_REBOA_selected_column"]
if "Rat ID" in df_subset.columns and "Rat ID" in df_raw.columns:
    df_corr = df_subset[["Rat ID"]].copy()
    df_corr["Rat ID"] = df_corr["Rat ID"].astype(str)
    df_corr["4hr_NDS"] = max_4hr_nds.values
    bp_cols = [c for c in ["MAP at the time of MinSBP", "SBP5minpostROSC"] if c in df_raw.columns]
    if bp_cols:
        df_raw_str = df_raw[["Rat ID"] + bp_cols].copy()
        df_raw_str["Rat ID"] = df_raw_str["Rat ID"].astype(str)
        for c in bp_cols:
            df_raw_str[c] = pd.to_numeric(df_raw_str[c], errors="coerce")
        df_corr = df_corr.merge(df_raw_str, on="Rat ID", how="left")
    else:
        for c in ["MAP at the time of MinSBP", "SBP5minpostROSC"]:
            df_corr[c] = np.nan
else:
    df_corr = pd.DataFrame({
        "4hr_NDS": max_4hr_nds.values,
        "MAP at the time of MinSBP": np.nan,
        "SBP5minpostROSC": np.nan,
    })

df_corr["4hr_NDS"] = pd.to_numeric(df_corr["4hr_NDS"], errors="coerce")

def plot_correlation_with_stats(ax, x, y, xlabel, ylabel, title):
    """Scatter plot with regression line, r and p in title or annotation."""
    mask = np.isfinite(x) & np.isfinite(y)
    x_ = np.asarray(x, dtype=float)[mask]
    y_ = np.asarray(y, dtype=float)[mask]
    if len(x_) < 3:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        return
    r, p = stats.pearsonr(x_, y_)
    ax.scatter(x_, y_, alpha=0.7, edgecolors="k", linewidths=0.5)
    # Regression line
    slope, intercept, _, _, _ = stats.linregress(x_, y_)
    x_line = np.linspace(x_.min(), x_.max(), 100)
    ax.plot(x_line, slope * x_line + intercept, "r-", lw=2, label=f"r = {r:.3f}, p = {p:.4f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\nr = {r:.3f}, p = {p:.4f}")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

if "MAP at the time of MinSBP" in df_corr.columns:
    plot_correlation_with_stats(
        axes[0],
        df_corr["4hr_NDS"],
        df_corr["MAP at the time of MinSBP"],
        "4hr NDS",
        "MAP at the time of MinSBP",
        "4hr NDS vs MAP at the time of MinSBP",
    )
else:
    axes[0].text(0.5, 0.5, "Column not found", ha="center", va="center", transform=axes[0].transAxes)

if "SBP5minpostROSC" in df_corr.columns:
    plot_correlation_with_stats(
        axes[1],
        df_corr["4hr_NDS"],
        df_corr["SBP5minpostROSC"],
        "4hr NDS",
        "SBP5minpostROSC",
        "4hr NDS vs SBP5minpostROSC",
    )
else:
    axes[1].text(0.5, 0.5, "Column not found", ha="center", va="center", transform=axes[1].transAxes)

plt.tight_layout()
plt.show()

# Print r and p for each
for col, label in [("MAP at the time of MinSBP", "MAP at the time of MinSBP"), ("SBP5minpostROSC", "SBP5minpostROSC")]:
    if col not in df_corr.columns:
        continue
    x = pd.to_numeric(df_corr["4hr_NDS"], errors="coerce")
    y = pd.to_numeric(df_corr[col], errors="coerce")
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() >= 3:
        r, p = stats.pearsonr(x[mask], y[mask])
        print(f"  4hr NDS vs {label}:  r = {r:.4f},  p = {p:.4f}  (n = {mask.sum()})")
    else:
        print(f"  4hr NDS vs {label}:  insufficient pairs (n = {mask.sum()})")
