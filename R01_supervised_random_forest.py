"""
R01 Supervised Random Forest – local run pipeline.
Pipeline: load -> preprocess (NDS) -> plot NDS distribution -> good/bad labels
          -> feature selection -> random forest classification.
All preprocessing artifacts are saved to disk.
Run with: conda activate eightsleep-ml && python R01_supervised_random_forest.py
"""

import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_curve, roc_auc_score
import joblib

#%% Imports & config (paths for local data and outputs)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
OUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# Local data: single Excel (.xlsx) with sheets matching the notebook
PATH_EXCEL = os.path.join(DATA_DIR, "data.xlsx")  # change filename if needed
# Sheet names match the notebook (Google Sheets / gspread) and expected Excel sheet names
SHEET_NAMES = ["Filtered_Subset_No_REBOA", "No_REBOA_selected_column", "cleaned_no_reboa"]

#%% Block 1: Load data from Excel (.xlsx) by sheet name
print("Block 1: Load data")
if not os.path.isfile(PATH_EXCEL):
    raise FileNotFoundError(
        f"Excel file not found: {PATH_EXCEL}. "
        f"Place a .xlsx file in {DATA_DIR} with sheets: {SHEET_NAMES}"
    )
dfs = {}
try:
    for sheet_name in SHEET_NAMES:
        dfs[sheet_name] = pd.read_excel(PATH_EXCEL, sheet_name=sheet_name, engine="openpyxl")
        print(f"  Loaded sheet '{sheet_name}': {len(dfs[sheet_name])} rows")
except Exception as e:
    raise FileNotFoundError(
        f"Failed to read Excel from {PATH_EXCEL}. "
        f"Ensure sheets exist: {SHEET_NAMES}. Error: {e}"
    ) from e

#%% Block 2: Preprocess NDS (extract max scores, 4hr / 24hr / 24morning)
print("Block 2: Preprocess NDS")


def extract_max_score(nds_string):
    """Extract the maximum number from strings like '32(40)' or '27'."""
    if isinstance(nds_string, str):
        numbers = re.findall(r"\d+", nds_string)
        if numbers:
            return max(map(int, numbers))
    return None


df_subset = dfs.get("Filtered_Subset_No_REBOA")
if df_subset is None:
    raise ValueError("Filtered_Subset_No_REBOA is required for NDS preprocessing.")

# 4hr NDS
nds_4hr_col = "4hr NDS" if "4hr NDS" in df_subset.columns else None
if nds_4hr_col:
    nds_4hr = df_subset[nds_4hr_col]
    max_4hr_nds_scores = nds_4hr.apply(extract_max_score)
else:
    max_4hr_nds_scores = pd.Series(dtype=float)

# 24hr NDS
nds_24hr_col = "24hr NDS" if "24hr NDS" in df_subset.columns else None
if nds_24hr_col:
    nds_24hr = df_subset[nds_24hr_col]
    max_nds_24hr_scores = nds_24hr.apply(extract_max_score)
    max_nds_24hr_scores = max_nds_24hr_scores.fillna(60)
else:
    max_nds_24hr_scores = pd.Series(dtype=float)

# 24morning NDS: use where > 24hr to update 24hr
nds_24morning_col = "24morning NDS" if "24morning NDS" in df_subset.columns else None
if nds_24morning_col and len(max_nds_24hr_scores) > 0:
    nds_24morning = df_subset[nds_24morning_col]
    max_nds_24morning_scores = nds_24morning.apply(extract_max_score)
    max_nds_24hr_scores = max_nds_24hr_scores.reindex(max_nds_24morning_scores.index)
    mask = max_nds_24morning_scores > max_nds_24hr_scores
    max_nds_24hr_scores = max_nds_24hr_scores.where(~mask, max_nds_24morning_scores)

# Save NDS preprocessing
joblib.dump(
    {
        "max_4hr_nds_scores": max_4hr_nds_scores,
        "max_nds_24hr_scores": max_nds_24hr_scores,
        "df_subset_index": df_subset.index.tolist(),
    },
    os.path.join(OUT_DIR, "preprocess_nds.joblib"),
)
max_4hr_nds_scores.to_csv(os.path.join(OUT_DIR, "max_4hr_nds_scores.csv"))
max_nds_24hr_scores.to_csv(os.path.join(OUT_DIR, "max_nds_24hr_scores.csv"))
print("  NDS preprocessing saved to outputs/")

#%% Block 3: Plot NDS distribution (4hr and 24hr)
print("Block 3: Plot NDS distribution")
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    valid_4hr = max_4hr_nds_scores.dropna().astype(float)
    if len(valid_4hr) > 0:
        axes[0].hist(valid_4hr, bins=15, edgecolor="black", alpha=0.7, label="_nolegend_")
        axes[0].set_title("4hr NDS (max score) distribution")
        axes[0].set_xlabel("NDS score")
        axes[0].set_ylabel("Count")
        med_4 = float(np.median(valid_4hr))
        mean_4 = float(np.mean(valid_4hr))
        q1_4, q3_4 = float(np.percentile(valid_4hr, 25)), float(np.percentile(valid_4hr, 75))
        iqr_4 = q3_4 - q1_4
        n_4 = len(valid_4hr)
        patch_4 = mpatches.Patch(visible=False)
        axes[0].legend(
            handles=[patch_4],
            labels=[f"Median: {med_4:.2f}\nMean: {mean_4:.2f}\nIQR: {iqr_4:.2f}\nn = {n_4}"],
            loc="upper right",
            framealpha=0.9,
            fontsize=9,
        )
    valid_24hr = max_nds_24hr_scores.dropna().astype(float)
    if len(valid_24hr) > 0:
        axes[1].hist(valid_24hr, bins=15, edgecolor="black", alpha=0.7, label="_nolegend_")
        axes[1].set_title("24hr NDS (max score) distribution")
        axes[1].set_xlabel("NDS score")
        axes[1].set_ylabel("Count")
        med_24 = float(np.median(valid_24hr))
        mean_24 = float(np.mean(valid_24hr))
        q1_24, q3_24 = float(np.percentile(valid_24hr, 25)), float(np.percentile(valid_24hr, 75))
        iqr_24 = q3_24 - q1_24
        n_24 = len(valid_24hr)
        patch_24 = mpatches.Patch(visible=False)
        axes[1].legend(
            handles=[patch_24],
            labels=[f"Median: {med_24:.2f}\nMean: {mean_24:.2f}\nIQR: {iqr_24:.2f}\nn = {n_24}"],
            loc="upper right",
            framealpha=0.9,
            fontsize=9,
        )
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "nds_distribution.png"), dpi=150)
    plt.close()
    print("  Saved outputs/nds_distribution.png")
except ModuleNotFoundError:
    print("  Skipped (matplotlib not installed). Install with: pip install matplotlib")

#%% Block 4: Define good/bad labels (single vs group percentile)
print("Block 4: Define good/bad labels (single vs group percentile)")


def classify_by_single_percentile(score_series, percentile_threshold=50):
    """Classify scores as 'good' or 'bad' by global percentile (single division)."""
    s = pd.Series(score_series)
    valid = s.dropna()
    if valid.empty:
        return np.full(len(s), np.nan, dtype=object)
    threshold = np.percentile(valid, percentile_threshold)
    out = np.where(s > threshold, "good", "bad")
    out = out.astype(object)
    out[s.isna().values] = np.nan
    return out


def classify_by_group_percentile(
    df,
    score_series,
    group_col="Intended Asphyxia/Asystole time (min)",
    percentile_threshold=50,
):
    """Classify scores as 'good' or 'bad' by percentile within group."""
    if len(df) != len(score_series):
        raise ValueError("Dataframe and score series must have the same length.")
    group_data = df[group_col]
    if isinstance(group_data, pd.DataFrame):
        group_data = group_data.iloc[:, 0]
    temp_df = pd.DataFrame(
        {group_col: group_data.values, "score": score_series.values}
    )
    temp_df["classification"] = np.nan
    temp_df["classification"] = temp_df["classification"].astype(object)
    for group in temp_df[group_col].unique():
        if pd.isna(group):
            group_indices = temp_df[group_col].isna()
        else:
            group_indices = temp_df[group_col] == group
        group_scores = temp_df.loc[group_indices, "score"]
        valid_scores = group_scores.dropna()
        if not valid_scores.empty:
            threshold = np.percentile(valid_scores, percentile_threshold)
            temp_df.loc[group_indices, "classification"] = np.where(
                group_scores > threshold, "good", "bad"
            )
        else:
            temp_df.loc[group_indices, "classification"] = "bad"
    return temp_df["classification"].values


if "Filtered_Subset_No_REBOA" in dfs and len(max_4hr_nds_scores) > 0:
    # Group division (percentile within each asphyxia group)
    classification_4hr_nds_grouped = classify_by_group_percentile(
        df_subset,
        max_4hr_nds_scores,
        group_col="Intended Asphyxia/Asystole time (min)",
        percentile_threshold=50,
    )
    classification_24hr_nds_grouped = classify_by_group_percentile(
        df_subset,
        max_nds_24hr_scores,
        group_col="Intended Asphyxia/Asystole time (min)",
        percentile_threshold=50,
    )
    # Single division (global percentile)
    classification_4hr_single = classify_by_single_percentile(
        max_4hr_nds_scores, percentile_threshold=50
    )
    classification_24hr_single = classify_by_single_percentile(
        max_nds_24hr_scores, percentile_threshold=50
    )
    pd.DataFrame(
        {
            "classification_4hr_group": classification_4hr_nds_grouped,
            "classification_24hr_group": classification_24hr_nds_grouped,
            "classification_4hr_single": classification_4hr_single,
            "classification_24hr_single": classification_24hr_single,
        }
    ).to_csv(os.path.join(OUT_DIR, "labels_good_bad.csv"), index=False)
    joblib.dump(
        {
            "classification_4hr_nds_grouped": classification_4hr_nds_grouped,
            "classification_24hr_nds_grouped": classification_24hr_nds_grouped,
            "classification_4hr_single": classification_4hr_single,
            "classification_24hr_single": classification_24hr_single,
        },
        os.path.join(OUT_DIR, "labels_good_bad.joblib"),
    )
    print("  Labels saved to outputs/labels_good_bad.csv and .joblib")

    # Visualize single vs group division label changes
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, single_arr, group_arr, title in [
            (axes[0], classification_4hr_single, classification_4hr_nds_grouped, "4hr NDS: Single vs group division"),
            (axes[1], classification_24hr_single, classification_24hr_nds_grouped, "24hr NDS: Single vs group division"),
        ]:
            single_arr = np.asarray(single_arr)
            group_arr = np.asarray(group_arr)
            valid = ~(pd.isna(single_arr) | pd.isna(group_arr))
            s, g = single_arr[valid], group_arr[valid]
            both_good = np.sum((s == "good") & (g == "good"))
            both_bad = np.sum((s == "bad") & (g == "bad"))
            single_good_group_bad = np.sum((s == "good") & (g == "bad"))
            single_bad_group_good = np.sum((s == "bad") & (g == "good"))
            labels = ["Both good", "Both bad", "Single good\nGroup bad", "Single bad\nGroup good"]
            counts = [both_good, both_bad, single_good_group_bad, single_bad_group_good]
            colors = ["C2", "C3", "C1", "C0"]
            bars = ax.bar(labels, counts, color=colors, edgecolor="black")
            ax.set_title(title)
            ax.set_ylabel("Count")
            for b in bars:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5, str(int(b.get_height())), ha="center", fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "label_single_vs_group_division.png"), dpi=150)
        plt.close()
        print("  Saved outputs/label_single_vs_group_division.png")
    except Exception as e:
        print("  Could not plot single vs group division:", e)
else:
    classification_4hr_nds_grouped = None
    classification_24hr_nds_grouped = None
    classification_4hr_single = None
    classification_24hr_single = None

#%% Block 5: Build feature table (No_REBOA_selected_column + DBP/SBP/MAP norm)
print("Block 5: Build feature table (normalize DBP/SBP/MAP)")

df_no_reboa = dfs.get("No_REBOA_selected_column")
if df_no_reboa is None:
    df_no_reboa = pd.DataFrame()

if len(df_no_reboa) > 0:
    df_no_reboa = df_no_reboa.copy()

    baseline_dbp_column = "DBP_baseline"
    dbp_columns_to_normalize = [
        "DBP start (Asphyxia/KCl)",
        "DBP EEGflat",
        "MaxDBP (postROSC)",
        "MinDBP (postROSC)",
        "DBP5minpostROSC",
        "DBP30minpostROSC",
        "DBP for MaxSBP",
        "DBP for MinSBP",
    ]
    if baseline_dbp_column in df_no_reboa.columns:
        for col in dbp_columns_to_normalize + [baseline_dbp_column]:
            if col in df_no_reboa.columns:
                df_no_reboa[col] = pd.to_numeric(df_no_reboa[col], errors="coerce")
        if df_no_reboa[baseline_dbp_column].isnull().any():
            m = df_no_reboa[baseline_dbp_column].mean()
            df_no_reboa[baseline_dbp_column] = df_no_reboa[baseline_dbp_column].fillna(m)
        for col in dbp_columns_to_normalize:
            if col in df_no_reboa.columns:
                df_no_reboa[col] = df_no_reboa[col] / df_no_reboa[baseline_dbp_column]

    baseline_sbp_column = "SBP_baseline"
    sbp_columns_to_normalize = [
        "SBP start (Asphyxia/KCl)",
        "SBP EEGflat",
        "MaxSBP (postROSC)",
        "MinSBP (postROSC)",
        "SBP for Min DBP",
        "SBP for Max DBP",
        "SBP5minpostROSC",
        "SBP30minpostROSC",
    ]
    if baseline_sbp_column in df_no_reboa.columns:
        for col in sbp_columns_to_normalize + [baseline_sbp_column]:
            if col in df_no_reboa.columns:
                df_no_reboa[col] = pd.to_numeric(df_no_reboa[col], errors="coerce")
        if df_no_reboa[baseline_sbp_column].isnull().any():
            m = df_no_reboa[baseline_sbp_column].mean()
            df_no_reboa[baseline_sbp_column] = df_no_reboa[baseline_sbp_column].fillna(m)
        for col in sbp_columns_to_normalize:
            if col in df_no_reboa.columns:
                df_no_reboa[col] = df_no_reboa[col] / df_no_reboa[baseline_sbp_column]

    baseline_map_column = "MAP_baseline"
    map_columns_to_normalize = [
        "MAP start (Asphyxia/KCl, whole integer)",
        "MAP EEG flat",
        "MAP EEG Flat Formula",
        "MAP for MaxSBP",
        "MAP at the time of MinSBP",
        "MAP at the time of MaxDBP",
        "MAP at the time of MinDBP",
    ]
    if baseline_map_column in df_no_reboa.columns:
        for col in map_columns_to_normalize + [baseline_map_column]:
            if col in df_no_reboa.columns:
                df_no_reboa[col] = pd.to_numeric(df_no_reboa[col], errors="coerce")
        if df_no_reboa[baseline_map_column].isnull().any():
            m = df_no_reboa[baseline_map_column].mean()
            df_no_reboa[baseline_map_column] = df_no_reboa[baseline_map_column].fillna(m)
        for col in map_columns_to_normalize:
            if col in df_no_reboa.columns:
                df_no_reboa[col] = df_no_reboa[col] / df_no_reboa[baseline_map_column]

    df_no_reboa.to_csv(os.path.join(OUT_DIR, "df_no_reboa_normalized.csv"), index=False)
    joblib.dump(
        {
            "dbp_columns_to_normalize": dbp_columns_to_normalize,
            "sbp_columns_to_normalize": sbp_columns_to_normalize,
            "map_columns_to_normalize": map_columns_to_normalize,
            "baseline_dbp_column": baseline_dbp_column,
            "baseline_sbp_column": baseline_sbp_column,
            "baseline_map_column": baseline_map_column,
        },
        os.path.join(OUT_DIR, "preprocess_bp_config.joblib"),
    )
    print("  df_no_reboa (normalized) and BP config saved to outputs/")

#%% Block 6: Feature selection (numeric matrix, drop ID and BSR column)
print("Block 6: Feature selection")

if len(df_no_reboa) == 0:
    df_no_reboa_numeric = pd.DataFrame()
else:
    df_no_reboa_numeric = df_no_reboa.apply(pd.to_numeric, errors="coerce").fillna(0)

drop_cols = ["Rat ID", "Time from ROSC that BSR Reaches 0.5"]
feature_cols = [c for c in df_no_reboa_numeric.columns if c not in drop_cols]
X = df_no_reboa_numeric[feature_cols].copy() if feature_cols else pd.DataFrame()

# Align labels with feature matrix by Rat ID (Filtered_Subset has labels, selected_column has features)
# Build y_4hr and y_24hr for both 4hr and 24hr NDS classification (group-based labels)
if classification_4hr_nds_grouped is not None and len(X) > 0 and "Rat ID" in df_no_reboa.columns:
    rat_id_feature = df_no_reboa["Rat ID"].astype(str).iloc[: len(X)].values
    rat_id_label = df_subset["Rat ID"].astype(str).values if "Rat ID" in df_subset.columns else None
    if rat_id_label is not None and len(rat_id_label) == len(classification_4hr_nds_grouped):
        label_df = pd.DataFrame(
            {
                "Rat ID": rat_id_label,
                "y_4hr": np.where(np.array(classification_4hr_nds_grouped) == "good", 1, 0),
                "y_24hr": np.where(np.array(classification_24hr_nds_grouped) == "good", 1, 0),
            }
        )
        merged = X.copy()
        merged["Rat ID"] = rat_id_feature
        merged = merged.merge(label_df, on="Rat ID", how="inner")
        y_4hr = merged["y_4hr"].values
        y_24hr = merged["y_24hr"].values
        X = merged.drop(columns=["Rat ID", "y_4hr", "y_24hr"])
    else:
        n_align = min(len(X), len(classification_4hr_nds_grouped))
        y_4hr = np.where(np.array(classification_4hr_nds_grouped[:n_align]) == "good", 1, 0)
        y_24hr = np.where(np.array(classification_24hr_nds_grouped[:n_align]) == "good", 1, 0)
        X = X.iloc[:n_align]
else:
    y_4hr = np.array([])
    y_24hr = np.array([])

np.save(os.path.join(OUT_DIR, "feature_names.npy"), np.array(X.columns.tolist()))
joblib.dump(
    {"feature_cols": feature_cols, "drop_cols": drop_cols},
    os.path.join(OUT_DIR, "feature_selection.joblib"),
)
print("  Feature matrix shape:", X.shape if len(X) > 0 else (0, 0))

#%% Block 7: Random forest classification (4hr and 24hr NDS) + ROC + SHAP
print("Block 7: Random forest classification (4hr and 24hr), ROC plots, SHAP")

def _train_eval_rf(X, y, target_name, random_state=42):
    """Train RF, return model, X_train, X_test, y_train, y_test, y_proba_test."""
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=random_state)
    clf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    y_proba = clf.predict_proba(X_te)[:, 1]  # P(good)
    acc = accuracy_score(y_te, y_pred)
    return clf, X_tr, X_te, y_tr, y_te, y_proba, acc, y_pred

models_to_run = []
if len(X) > 0 and len(y_4hr) > 0 and np.unique(y_4hr).size > 1:
    models_to_run.append(("4hr NDS", y_4hr))
if len(X) > 0 and len(y_24hr) > 0 and np.unique(y_24hr).size > 1:
    models_to_run.append(("24hr NDS", y_24hr))

if models_to_run:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        plt = None
    n_models = len(models_to_run)
    results = {}
    for target_name, y in models_to_run:
        model, X_train, X_test, y_train, y_test, y_proba, acc, y_pred = _train_eval_rf(
            X, y, target_name
        )
        results[target_name] = {
            "model": model,
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "y_proba": y_proba,
            "accuracy": acc,
            "y_pred": y_pred,
        }
        print(f"  [{target_name}] Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred, target_names=["bad", "good"]))
        importance_df = pd.DataFrame(
            {"Feature": X.columns, "Importance": model.feature_importances_}
        ).sort_values(by="Importance", ascending=False)
        importance_df.to_csv(
            os.path.join(OUT_DIR, f"feature_importance_{target_name.replace(' ', '_')}.csv"),
            index=False,
        )
        joblib.dump(model, os.path.join(OUT_DIR, f"random_forest_{target_name.replace(' ', '_')}.joblib"))

    # ROC curves (plot only, do not save)
    if plt is not None and n_models > 0:
        fig_roc, axes_roc = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
        if n_models == 1:
            axes_roc = [axes_roc]
        for ax, (target_name, res) in zip(axes_roc, results.items()):
            fpr, tpr, _ = roc_curve(res["y_test"], res["y_proba"])
            auc = roc_auc_score(res["y_test"], res["y_proba"])
            ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
            ax.plot([0, 1], [0, 1], "k--")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title(f"ROC — {target_name}")
            ax.legend()
            ax.set_aspect("equal")
        plt.tight_layout()
        plt.show()
        plt.close()

    # SHAP analysis
    try:
        import shap
    except ImportError:
        shap = None
    if shap is not None:
        for target_name, res in results.items():
            model = res["model"]
            X_train = res["X_train"]
            # Use a sample if large to keep SHAP fast
            X_shap = X_train if len(X_train) <= 200 else X_train.sample(200, random_state=42)
            explainer = shap.TreeExplainer(model, X_shap)
            shap_values = explainer.shap_values(X_shap)
            # Binary classification: shap_values can be list [class0, class1] or single array
            if isinstance(shap_values, list):
                shap_vals = shap_values[1]  # use positive class (good)
            else:
                shap_vals = shap_values
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_vals, X_shap, show=False, max_display=20)
            plt.title(f"SHAP summary — {target_name}")
            plt.tight_layout()
            plt.savefig(
                os.path.join(OUT_DIR, f"shap_summary_{target_name.replace(' ', '_')}.png"),
                dpi=150,
                bbox_inches="tight",
            )
            plt.close()
            print(f"  SHAP summary saved: outputs/shap_summary_{target_name.replace(' ', '_')}.png")
    else:
        print("  SHAP skipped (pip install shap for SHAP analysis).")

    joblib.dump(
        {
            "X_train_4hr": results.get("4hr NDS", {}).get("X_train"),
            "X_test_4hr": results.get("4hr NDS", {}).get("X_test"),
            "y_train_4hr": results.get("4hr NDS", {}).get("y_train"),
            "y_test_4hr": results.get("4hr NDS", {}).get("y_test"),
            "X_train_24hr": results.get("24hr NDS", {}).get("X_train"),
            "X_test_24hr": results.get("24hr NDS", {}).get("X_test"),
            "y_train_24hr": results.get("24hr NDS", {}).get("y_train"),
            "y_test_24hr": results.get("24hr NDS", {}).get("y_test"),
        },
        os.path.join(OUT_DIR, "train_test_split.joblib"),
    )
    print("  Models and feature importance saved to outputs/")
else:
    print("  Skipped (insufficient data or single class in both targets).")

#%% Save full preprocessing summary (all artifacts referenced)
joblib.dump(
    {
        "data_paths": {"excel": PATH_EXCEL, "sheet_names": SHEET_NAMES},
        "out_dir": OUT_DIR,
        "n_samples": len(df_no_reboa) if len(df_no_reboa) > 0 else 0,
        "n_features": X.shape[1] if len(X) > 0 else 0,
    },
    os.path.join(OUT_DIR, "preprocessing_summary.joblib"),
)
print("Done. All preprocessing and outputs in:", OUT_DIR)
