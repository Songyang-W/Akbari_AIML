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
    from ml_pipeline import run_nested_cv, summarize_results, plot_comparison

    results = run_nested_cv(X, y)
    summarize_results(results, target_name="4hr NDS")
    plot_comparison(results, target_name="4hr NDS")
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


# ------------------------------------------------------------------ #
#  Nested cross-validation                                            #
# ------------------------------------------------------------------ #
def run_nested_cv(
    X, y, *,
    corr_threshold=0.85,
    outer_k=5,
    inner_k=5,
    random_state=42,
):
    """Nested stratified CV comparing three classifiers.

    Inner loop  — ``GridSearchCV`` (hyperparam tuning + feature selection).
    Outer loop  — unbiased ROC-AUC on held-out fold.

    Parameters
    ----------
    X : array-like  (n_samples, n_features)
    y : array-like  (n_samples,)  binary 0/1
    corr_threshold : float   |r| cutoff for CorrelationPruner
    outer_k, inner_k : int   number of CV folds
    random_state : int

    Returns
    -------
    results : dict  ``{model_name: {"scores", "mean", "std", "best_params"}}``
    """
    X_arr = np.asarray(X, dtype=float)
    y_arr = np.asarray(y, dtype=int)

    if len(np.unique(y_arr)) < 2:
        raise ValueError("Target has only one class — cannot run CV.")

    min_class = int(np.min(np.bincount(y_arr)))
    if min_class < outer_k:
        outer_k = max(2, min_class)
        inner_k = max(2, min_class - 1)
        print(f"  [WARNING] Minority class has {min_class} samples "
              f"— reducing to outer_k={outer_k}, inner_k={inner_k}")

    outer_cv = StratifiedKFold(
        n_splits=outer_k, shuffle=True, random_state=random_state,
    )
    inner_cv = StratifiedKFold(
        n_splits=inner_k, shuffle=True, random_state=random_state,
    )

    configs = _build_model_configs(corr_threshold, random_state)
    results = {}

    for name, cfg in configs.items():
        print(f"\n  [{name}]  {outer_k}-fold outer × {inner_k}-fold inner")
        fold_scores, fold_params = [], []

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
            print(f"    Fold {fold_i + 1}: AUC = {auc:.4f}")

        results[name] = {
            "scores": fold_scores,
            "mean":   float(np.mean(fold_scores)),
            "std":    float(np.std(fold_scores)),
            "best_params": fold_params,
        }
        print(f"  [{name}]  >>>  AUC = {results[name]['mean']:.4f} "
              f"± {results[name]['std']:.4f}")

    return results


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
