"""
ml_pipeline.py — Nested-CV model comparison with feature selection.

Each model's sklearn Pipeline (fitted per CV fold):
    CorrelationPruner → StandardScaler → SelectFromModel → Classifier

Models compared:
    1. Elastic-Net Logistic Regression
    2. Random Forest
    3. XGBoost  (gracefully skipped if not installed)

Usage
-----
    # Single run (verbose)
    from ml_pipeline import run_nested_cv, summarize_results, plot_comparison, report_feature_selection
    results = run_nested_cv(X, y, feature_names=X.columns.tolist(), return_feature_selection=True)
    summarize_results(results, target_name="4hr NDS")
    report_feature_selection(results)
    plot_comparison(results, target_name="4hr NDS")

    # Repeated over seeds (compact output)
    from ml_pipeline import run_repeated_nested_cv, print_repeated_cv_summary, plot_comparison
    out = run_repeated_nested_cv(X, y, seeds=(11,22,33,44,55), verbose=False)
    print_repeated_cv_summary(out, target_name="4hr NDS", min_freq=0.80)
    plot_comparison({m: {"scores": out["raw_scores"][m]} for m in out["raw_scores"]}, target_name="4hr NDS")
"""

import warnings
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score


# ------------------------------------------------------------------ #
#  Custom transformer: correlation-based feature pruning              #
# ------------------------------------------------------------------ #
class CorrelationPruner(BaseEstimator, TransformerMixin):
    """Drop features whose pairwise |r| exceeds *threshold*.

    For every pair above the cutoff the *later* column (by position)
    is removed.  Fitted on training data only, then applied to test.
    """

    def __init__(self, threshold=0.85):
        self.threshold = threshold

    def fit(self, X, y=None):
        corr = pd.DataFrame(X).corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
        to_drop = set()
        for col in upper.columns:
            if (upper[col] > self.threshold).any():
                to_drop.add(col)
        self.keep_mask_ = np.array(
            [i not in to_drop for i in range(X.shape[1])]
        )
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.keep_mask_]


# ------------------------------------------------------------------ #
#  Pipeline / param-grid factory                                      #
# ------------------------------------------------------------------ #
def _build_model_configs(corr_threshold, random_state):
    """Return ``{name: {"pipe": Pipeline, "grid": dict}}``."""

    rs = random_state
    configs = {}

    # --- Elastic-Net Logistic ---
    configs["ElasticNet_Logistic"] = {
        "pipe": Pipeline([
            ("pruner", CorrelationPruner(threshold=corr_threshold)),
            ("scaler", StandardScaler()),
            ("selector", SelectFromModel(
                LogisticRegression(
                    penalty="l1", solver="saga", max_iter=5000, random_state=rs,
                ),
                threshold="median",
            )),
            ("clf", LogisticRegression(
                penalty="elasticnet", solver="saga", max_iter=5000, random_state=rs,
            )),
        ]),
        "grid": {
            "clf__C":        [0.01, 0.1, 1.0, 10.0],
            "clf__l1_ratio": [0.1, 0.5, 0.9],
        },
    }

    # --- Random Forest ---
    configs["Random_Forest"] = {
        "pipe": Pipeline([
            ("pruner", CorrelationPruner(threshold=corr_threshold)),
            ("scaler", StandardScaler()),
            ("selector", SelectFromModel(
                RandomForestClassifier(
                    n_estimators=50, random_state=rs,
                ),
                threshold="median",
            )),
            ("clf", RandomForestClassifier(random_state=rs)),
        ]),
        "grid": {
            "clf__n_estimators":    [100, 200],
            "clf__max_depth":       [3, 5, None],
            "clf__min_samples_leaf": [1, 5],
        },
    }

    # --- XGBoost (optional) ---
    try:
        from xgboost import XGBClassifier

        configs["XGBoost"] = {
            "pipe": Pipeline([
                ("pruner", CorrelationPruner(threshold=corr_threshold)),
                ("scaler", StandardScaler()),
                ("selector", SelectFromModel(
                    XGBClassifier(
                        n_estimators=50, verbosity=0, random_state=rs,
                    ),
                    threshold="median",
                )),
                ("clf", XGBClassifier(verbosity=0, random_state=rs)),
            ]),
            "grid": {
                "clf__n_estimators": [100, 200],
                "clf__max_depth":    [3, 5],
                "clf__learning_rate": [0.01, 0.1],
            },
        }
    except ImportError:
        print("  [WARNING] xgboost not installed — skipping.  pip install xgboost")

    return configs


def _build_model_configs_no_fs(random_state):
    """Pipelines with no feature selection: StandardScaler + classifier only."""
    rs = random_state
    configs = {}

    configs["ElasticNet_Logistic"] = {
        "pipe": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                penalty="elasticnet", solver="saga", max_iter=5000, random_state=rs,
            )),
        ]),
        "grid": {
            "clf__C":        [0.01, 0.1, 1.0, 10.0],
            "clf__l1_ratio": [0.1, 0.5, 0.9],
        },
    }

    configs["Random_Forest"] = {
        "pipe": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(random_state=rs)),
        ]),
        "grid": {
            "clf__n_estimators":    [100, 200],
            "clf__max_depth":       [3, 5, None],
            "clf__min_samples_leaf": [1, 5],
        },
    }

    try:
        from xgboost import XGBClassifier
        configs["XGBoost"] = {
            "pipe": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", XGBClassifier(verbosity=0, random_state=rs)),
            ]),
            "grid": {
                "clf__n_estimators": [100, 200],
                "clf__max_depth":    [3, 5],
                "clf__learning_rate": [0.01, 0.1],
            },
        }
    except ImportError:
        pass

    return configs


# ------------------------------------------------------------------ #
#  Nested cross-validation                                            #
# ------------------------------------------------------------------ #
def _extract_feature_selection(pipeline, feature_names):
    """From a fitted pipeline, return dropped/selected feature names."""
    pruner = pipeline.named_steps["pruner"]
    selector = pipeline.named_steps["selector"]

    # After correlation pruning
    kept_after_prune = [feature_names[i] for i in range(len(feature_names))
                        if pruner.keep_mask_[i]]
    dropped_by_corr = [feature_names[i] for i in range(len(feature_names))
                       if not pruner.keep_mask_[i]]

    # After embedded selection (selector operates on pruned features)
    sel_support = selector.get_support()
    selected = [kept_after_prune[i] for i in range(len(kept_after_prune))
                if sel_support[i]]
    dropped_by_sel = [kept_after_prune[i] for i in range(len(kept_after_prune))
                      if not sel_support[i]]

    return {
        "dropped_by_correlation": dropped_by_corr,
        "dropped_by_selector": dropped_by_sel,
        "selected": selected,
        "n_selected": len(selected),
    }


def run_nested_cv(
    X, y, *,
    corr_threshold=0.85,
    outer_k=5,
    inner_k=5,
    random_state=42,
    feature_names=None,
    return_feature_selection=False,
    verbose=True,
    feature_selection=True,
):
    """Nested stratified CV comparing three classifiers.

    Inner loop  — ``GridSearchCV`` (hyperparam tuning; + feature selection if feature_selection=True).
    Outer loop  — unbiased ROC-AUC on held-out fold.

    Parameters
    ----------
    X : array-like or DataFrame  (n_samples, n_features)
    y : array-like  (n_samples,)  binary 0/1
    corr_threshold : float   |r| cutoff for CorrelationPruner (ignored if feature_selection=False)
    outer_k, inner_k : int   number of CV folds
    random_state : int
    feature_names : list, optional   names for each column (inferred from X if DataFrame)
    return_feature_selection : bool   if True and feature_selection=True, store per-fold dropped/selected features
    verbose : bool   if True, print per-fold and per-model lines
    feature_selection : bool   if False, use all features (no pruner, no selector)

    Returns
    -------
    results : dict  ``{model_name: {"scores", "mean", "std", "best_params",
                  "feature_selection": [...]}}``  (feature_selection if requested and feature_selection=True)
    """
    if hasattr(X, "columns"):
        X_arr = np.asarray(X, dtype=float)
        fnames = feature_names if feature_names is not None else X.columns.tolist()
    else:
        X_arr = np.asarray(X, dtype=float)
        fnames = feature_names

    if feature_selection and return_feature_selection and fnames is None:
        return_feature_selection = False
        if verbose:
            print("  [WARNING] feature_names not provided — skipping feature selection tracking")
    if not feature_selection:
        return_feature_selection = False

    y_arr = np.asarray(y, dtype=int)

    if len(np.unique(y_arr)) < 2:
        raise ValueError("Target has only one class — cannot run CV.")

    min_class = int(np.min(np.bincount(y_arr)))
    if min_class < outer_k:
        outer_k = max(2, min_class)
        inner_k = max(2, min_class - 1)
        if verbose:
            print(f"  [WARNING] Minority class has {min_class} samples "
                  f"— reducing to outer_k={outer_k}, inner_k={inner_k}")

    outer_cv = StratifiedKFold(
        n_splits=outer_k, shuffle=True, random_state=random_state,
    )
    inner_cv = StratifiedKFold(
        n_splits=inner_k, shuffle=True, random_state=random_state,
    )

    configs = _build_model_configs(corr_threshold, random_state) if feature_selection else _build_model_configs_no_fs(random_state)
    results = {}

    for name, cfg in configs.items():
        if verbose:
            print(f"\n  [{name}]  {outer_k}-fold outer × {inner_k}-fold inner")
        fold_scores, fold_params = [], []
        fold_feat_sel = [] if (return_feature_selection and feature_selection) else None

        for fold_i, (tr_idx, te_idx) in enumerate(
            outer_cv.split(X_arr, y_arr)
        ):
            X_tr, X_te = X_arr[tr_idx], X_arr[te_idx]
            y_tr, y_te = y_arr[tr_idx], y_arr[te_idx]

            grid = GridSearchCV(
                clone(cfg["pipe"]),
                cfg["grid"],
                cv=inner_cv,
                scoring="roc_auc",
                n_jobs=-1,
                refit=True,
                error_score=0.0,
            )

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                grid.fit(X_tr, y_tr)

            y_proba = grid.predict_proba(X_te)[:, 1]
            auc = roc_auc_score(y_te, y_proba)
            fold_scores.append(auc)
            fold_params.append(grid.best_params_)
            if verbose:
                print(f"    Fold {fold_i + 1}: AUC = {auc:.4f}")

            if fold_feat_sel is not None:
                fs = _extract_feature_selection(grid.best_estimator_, fnames)
                fs["fold"] = fold_i + 1
                fold_feat_sel.append(fs)

        results[name] = {
            "scores": fold_scores,
            "mean":   float(np.mean(fold_scores)),
            "std":    float(np.std(fold_scores)),
            "best_params": fold_params,
        }
        if fold_feat_sel is not None:
            results[name]["feature_selection"] = fold_feat_sel
        if verbose:
            print(f"  [{name}]  >>>  AUC = {results[name]['mean']:.4f} "
                  f"± {results[name]['std']:.4f}")

    return results


def run_repeated_nested_cv(
    X, y, *,
    seeds=(11, 22, 33, 44, 55),
    outer_k=5,
    inner_k=5,
    corr_threshold=0.85,
    feature_names=None,
    verbose=False,
    feature_selection=True,
):
    """Repeated nested CV over multiple seeds; aggregate AUCs and (if feature_selection) feature frequencies.

    Parameters
    ----------
    X : array-like or DataFrame
    y : array-like  binary 0/1
    seeds : sequence of int   random seeds for outer/inner CV
    outer_k, inner_k : int
    corr_threshold : float   ignored if feature_selection=False
    feature_names : list, optional   inferred from X if DataFrame
    verbose : bool   if True, print per-fold details from each run_nested_cv
    feature_selection : bool   if False, no correlation/selector steps; feature_frequencies will be empty

    Returns
    -------
    dict with keys:
      raw_scores : {model_name: list of all outer-fold AUCs}
      seedwise_means : {model_name: list of mean AUC per seed}
      feature_frequencies : {model_name: {"selected", "corr_dropped", "selector_dropped"}} or empty
      n_folds : int   total outer folds
      summary_table : list of dicts [{"model", "n", "mean", "std", "median", "min", "max"}, ...] sorted by mean desc
    """
    if hasattr(X, "columns"):
        fnames = feature_names if feature_names is not None else X.columns.tolist()
    else:
        fnames = feature_names

    all_scores = {}
    seedwise_means = {}
    feature_frequencies = {}
    n_folds_total = 0

    for seed in seeds:
        res = run_nested_cv(
            X, y,
            corr_threshold=corr_threshold,
            outer_k=outer_k,
            inner_k=inner_k,
            random_state=seed,
            feature_names=fnames,
            return_feature_selection=feature_selection,
            verbose=verbose,
            feature_selection=feature_selection,
        )
        for name, r in res.items():
            all_scores.setdefault(name, []).extend(r["scores"])
            seedwise_means.setdefault(name, []).append(r["mean"])
            if "feature_selection" not in r:
                continue
            ff = feature_frequencies.setdefault(name, {
                "selected": {}, "corr_dropped": {}, "selector_dropped": {},
            })
            for fs in r["feature_selection"]:
                for feat in fs["selected"]:
                    ff["selected"][feat] = ff["selected"].get(feat, 0) + 1
                for feat in fs["dropped_by_correlation"]:
                    ff["corr_dropped"][feat] = ff["corr_dropped"].get(feat, 0) + 1
                for feat in fs["dropped_by_selector"]:
                    ff["selector_dropped"][feat] = ff["selector_dropped"].get(feat, 0) + 1

    n_folds_total = len(all_scores[list(all_scores.keys())[0]]) if all_scores else 0

    # summary_table: sort by mean AUC descending
    summary_table = []
    for name, scores in all_scores.items():
        scores_arr = np.array(scores)
        summary_table.append({
            "model": name,
            "n": len(scores),
            "mean": float(np.mean(scores_arr)),
            "std": float(np.std(scores_arr)),
            "median": float(np.median(scores_arr)),
            "min": float(np.min(scores_arr)),
            "max": float(np.max(scores_arr)),
        })
    summary_table.sort(key=lambda r: r["mean"], reverse=True)

    return {
        "raw_scores": all_scores,
        "seedwise_means": seedwise_means,
        "feature_frequencies": feature_frequencies,
        "n_folds": n_folds_total,
        "summary_table": summary_table,
    }


def print_repeated_cv_summary(
    repeated_result,
    target_name="",
    min_freq=0.80,
):
    """Print compact summary from run_repeated_nested_cv output."""
    table = repeated_result["summary_table"]
    ff = repeated_result["feature_frequencies"]
    n_folds = repeated_result["n_folds"]

    if target_name:
        print(f"Target: {target_name}")
    for row in table:
        name = row["model"]
        print(f"{name:<22} n={row['n']}  AUC={row['mean']:.3f}±{row['std']:.3f}  "
              f"med={row['median']:.3f}  min={row['min']:.3f}  max={row['max']:.3f}")
    print()

    for row in table:
        name = row["model"]
        if name not in ff:
            continue
        buckets = [
            ("selected", "selected"),
            ("corr_dropped", "corr_drop"),
            ("selector_dropped", "selector_drop"),
        ]
        for key, label in buckets:
            counts = ff[name].get(key, {})
            if not counts:
                continue
            items = sorted(counts.items(), key=lambda x: -x[1])
            above = [(f, c) for f, c in items if c >= min_freq * n_folds]
            if not above:
                continue
            parts = [f"{f}({c}/{n_folds})" for f, c in above]
            print(f"{name:<22} {label}>={min_freq:.2f}: {', '.join(parts)}")


def report_feature_selection(results, model_name=None):
    """Print which features were dropped/selected per fold.

    Call after run_nested_cv(..., return_feature_selection=True).

    Parameters
    ----------
    results : dict   from run_nested_cv
    model_name : str, optional   report only this model; if None, report all
    """
    models = [model_name] if model_name else list(results.keys())
    for name in models:
        if name not in results or "feature_selection" not in results[name]:
            print(f"  [{name}] No feature selection info (run with return_feature_selection=True)")
            continue
        fs_list = results[name]["feature_selection"]
        print(f"\n  --- {name} ---")
        for fs in fs_list:
            n_corr = len(fs["dropped_by_correlation"])
            n_sel = len(fs["dropped_by_selector"])
            n_kept = fs["n_selected"]
            print(f"    Fold {fs['fold']}: {n_kept} selected | "
                  f"dropped by correlation: {n_corr} | by selector: {n_sel}")
            if fs["dropped_by_correlation"]:
                print(f"      Dropped (corr): {fs['dropped_by_correlation']}")
            if fs["dropped_by_selector"]:
                print(f"      Dropped (selector): {fs['dropped_by_selector']}")
            print(f"      Selected: {fs['selected']}")


# ------------------------------------------------------------------ #
#  Reporting helpers                                                   #
# ------------------------------------------------------------------ #
def summarize_results(results, target_name=""):
    """Print a compact comparison table."""
    header = f"Model Comparison — {target_name}" if target_name else "Model Comparison"
    print(f"\n{'=' * 60}")
    print(header)
    print(f"{'=' * 60}")
    print(f"  {'Model':<22s}  {'Mean AUC':>9s}  {'± Std':>7s}  {'K':>3s}")
    print(f"  {'-' * 22}  {'-' * 9}  {'-' * 7}  {'-' * 3}")
    for name, res in results.items():
        k = len(res["scores"])
        print(f"  {name:<22s}  {res['mean']:9.4f}  {res['std']:7.4f}  {k:3d}")
    print()


def plot_comparison(results, target_name=""):
    """Box-and-strip plot of outer-fold AUC scores per model."""
    import matplotlib.pyplot as plt

    names = list(results.keys())
    data = [results[n]["scores"] for n in names]

    fig, ax = plt.subplots(figsize=(7, 4))
    bp = ax.boxplot(data, tick_labels=names, patch_artist=True, widths=0.4)

    palette = ["#6BAED6", "#74C476", "#FD8D3C"]
    for patch, color in zip(bp["boxes"], palette[: len(names)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    rng = np.random.default_rng(42)
    for i, scores in enumerate(data):
        jitter = rng.uniform(-0.06, 0.06, size=len(scores))
        ax.scatter(
            np.full(len(scores), i + 1) + jitter, scores,
            color="black", s=30, zorder=3, alpha=0.7,
        )

    title = f"Nested-CV AUC — {target_name}" if target_name else "Nested-CV AUC"
    ax.set_title(title, fontweight="bold")
    ax.set_ylabel("ROC AUC (outer fold)")
    ax.set_ylim(-0.05, 1.10)
    plt.tight_layout()
    plt.show()
