import json
import pickle
import signal
import sys
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", message=".*eval_set.*")

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from preprocess import ROOT, build_features


OUT = ROOT / "results" / "model"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
EARLY_STOPPING = 100

PARAMS = dict(
    objective="binary",
    n_estimators=5000,
    learning_rate=0.03,
    num_leaves=31,
    max_depth=-1,
    min_child_samples=100,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.4,
    reg_alpha=1.0,
    reg_lambda=10.0,
    min_split_gain=0.0,
    random_state=SEED,
    n_jobs=-1,
    verbose=-1,
)


def stop_training(signum, frame):
    print("\nTraining interrupted by user.")
    raise KeyboardInterrupt


signal.signal(signal.SIGINT, stop_training)


def feature_cols(df):
    return [
        col
        for col in df.columns
        if col not in ("SK_ID_CURR", "TARGET")
    ]


def train_model(X_train, y_train, n_estimators):
    params = {
        **PARAMS,
        "n_estimators": n_estimators,
    }

    return lgb.LGBMClassifier(**params).fit(
        X_train,
        y_train,
    )


def train_model_with_early_stopping(
    X_train,
    y_train,
    X_valid,
    y_valid,
):
    model = lgb.LGBMClassifier(**PARAMS)

    model.fit(
        X_train,
        y_train,
        eval_set=[
            (X_train, y_train),
            (X_valid, y_valid),
        ],
        eval_names=["train", "valid"],
        eval_metric="auc",
        callbacks=[
            lgb.early_stopping(
                EARLY_STOPPING,
                first_metric_only=False,
                verbose=False,
            ),
            lgb.log_evaluation(100),
        ],
    )

    return model


def main():
    start_time = time.time()

    try:
        print("Building features...")
        train_df, test_df = build_features()

        features = feature_cols(train_df)

        print(
            f"{len(train_df):,} labelled clients | "
            f"{len(features)} features | "
            f"default rate: {train_df.TARGET.mean():.3%}"
        )

        print("Creating train / holdout split...")

        train_part, holdout = train_test_split(
            train_df,
            test_size=0.20,
            stratify=train_df.TARGET,
            random_state=SEED,
        )

        holdout[["SK_ID_CURR"]].to_csv(
            OUT / "holdout_ids.csv",
            index=False,
        )

        print("Creating validation split...")

        fit, valid = train_test_split(
            train_part,
            test_size=0.20,
            stratify=train_part.TARGET,
            random_state=SEED,
        )

        X_fit = fit[features]
        y_fit = fit.TARGET

        X_valid = valid[features]
        y_valid = valid.TARGET

        print("Training LightGBM...")

        model = train_model_with_early_stopping(
            X_fit,
            y_fit,
            X_valid,
            y_valid,
        )

        best_iteration = (
            model.best_iteration_
            or PARAMS["n_estimators"]
        )

        print(f"Best iteration: {best_iteration}")

        evals = model.evals_result_

        p_fit = model.predict_proba(
            X_fit,
            num_iteration=best_iteration,
        )[:, 1]

        p_valid = model.predict_proba(
            X_valid,
            num_iteration=best_iteration,
        )[:, 1]

        p_holdout = model.predict_proba(
            holdout[features],
            num_iteration=best_iteration,
        )[:, 1]

        auc_fit = roc_auc_score(
            y_fit,
            p_fit,
        )

        auc_valid = roc_auc_score(
            y_valid,
            p_valid,
        )

        auc_holdout = roc_auc_score(
            holdout.TARGET,
            p_holdout,
        )

        print(
            f"AUC train: {auc_fit:.4f} | "
            f"valid: {auc_valid:.4f} | "
            f"holdout: {auc_holdout:.4f}"
        )

        print("Saving learning curve...")

        plot_iteration_curve(
            evals,
            best_iteration,
        )

        print("Training final model...")

        final = train_model(
            train_part[features],
            train_part.TARGET,
            best_iteration,
        )

        p_holdout = final.predict_proba(
            holdout[features]
        )[:, 1]

        auc_final = roc_auc_score(
            holdout.TARGET,
            p_holdout,
        )

        pr_auc = average_precision_score(
            holdout.TARGET,
            p_holdout,
        )

        brier = brier_score_loss(
            holdout.TARGET,
            p_holdout,
        )

        trivial_accuracy = 1 - holdout.TARGET.mean()

        print(
            f"FINAL | "
            f"AUC: {auc_final:.4f} | "
            f"PR-AUC: {pr_auc:.4f} | "
            f"Brier: {brier:.4f}"
        )

        print("Calculating feature importance...")

        importance = get_feature_importance(
            final,
            features,
        )

        plot_feature_importance(
            importance,
            len(features),
        )

        save_model(
            final,
            features,
            best_iteration,
        )

        metrics = {
            "n_features": len(features),
            "best_iteration": int(best_iteration),
            "auc_train": auc_fit,
            "auc_valid": auc_valid,
            "auc_holdout": auc_final,
            "pr_auc_holdout": pr_auc,
            "brier_holdout": brier,
            "trivial_accuracy": trivial_accuracy,
            "default_rate": float(
                train_df.TARGET.mean()
            ),
            "train_minutes": round(
                (time.time() - start_time) / 60,
                1,
            ),
        }

        metrics = {
            key: (
                float(value)
                if isinstance(value, (float, np.floating))
                else value
            )
            for key, value in metrics.items()
        }

        with open(OUT / "metrics.json", "w") as f:
            json.dump(
                metrics,
                f,
                indent=2,
            )

        write_report(
            metrics,
            importance,
        )

        print(
            f"Done in {metrics['train_minutes']} min"
        )

    except KeyboardInterrupt:
        print("\nTraining stopped cleanly.")
        sys.exit(130)


def plot_iteration_curve(evals, best_iteration):
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        evals["train"]["auc"],
        label="train",
    )

    ax.plot(
        evals["valid"]["auc"],
        label="validation",
    )

    ax.axvline(
        best_iteration,
        color="k",
        linestyle="--",
        label=f"best = {best_iteration}",
    )

    ax.set(
        xlabel="Boosting iterations",
        ylabel="AUC",
        title="Learning Curve",
    )

    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        OUT / "learning_curve_iterations.png",
        dpi=130,
    )

    plt.close(fig)


def get_feature_importance(model, features):
    importance = pd.DataFrame(
        {
            "feature": features,
            "gain": model.booster_.feature_importance(
                "gain"
            ),
            "split": model.booster_.feature_importance(
                "split"
            ),
        }
    )

    importance["gain_pct"] = (
        100
        * importance.gain
        / importance.gain.sum()
    )

    importance = (
        importance
        .sort_values("gain", ascending=False)
        .reset_index(drop=True)
    )

    importance.to_csv(
        OUT / "feature_importance.csv",
        index=False,
    )

    return importance


def plot_feature_importance(importance, n_features):
    top = importance.head(40).iloc[::-1]

    fig, ax = plt.subplots(figsize=(9, 11))

    ax.barh(
        top.feature,
        top.gain_pct,
    )

    ax.set(
        xlabel="% of total gain",
        title=f"Top 40 / {n_features} Features",
    )

    fig.tight_layout()

    fig.savefig(
        OUT / "feature_importance.png",
        dpi=130,
    )

    plt.close(fig)


def save_model(model, features, best_iteration):
    with open(
        OUT / "my_own_model.pkl",
        "wb",
    ) as f:
        pickle.dump(
            {
                "model": model,
                "features": features,
                "best_iteration": best_iteration,
                "params": {
                    **PARAMS,
                    "n_estimators": best_iteration,
                },
            },
            f,
        )


def write_report(metrics, importance):
    gap = (
        metrics["auc_train"]
        - metrics["auc_valid"]
    )

    top_features = "\n".join(
        f"{i + 1:2d}. {row.feature:<40} "
        f"{row.gain_pct:5.2f}%"
        for i, row in importance.head(10).iterrows()
    )

    report = f"""
HOME CREDIT — LIGHTGBM MODEL REPORT

===================================

MODEL
-----

Algorithm       : LightGBM
Features        : {metrics["n_features"]}
Best iteration  : {metrics["best_iteration"]}

PERFORMANCE
-----------

Train AUC       : {metrics["auc_train"]:.4f}
Validation AUC  : {metrics["auc_valid"]:.4f}
Holdout AUC     : {metrics["auc_holdout"]:.4f}
Holdout PR-AUC  : {metrics["pr_auc_holdout"]:.4f}
Brier score     : {metrics["brier_holdout"]:.4f}

OVERFITTING
-----------

Train-valid gap : {gap:.4f}

The holdout set was kept separate from model fitting
and early stopping.

TOP 10 FEATURES
---------------

{top_features}

LIMITATIONS
-----------

- Single train/holdout split.
- No out-of-time validation.
- Probabilities are not calibrated.
- Fairness and regulatory checks are required before real-world use.

TRAINING TIME
-------------

{metrics["train_minutes"]} minutes
""".strip()

    (OUT / "model_report.txt").write_text(
        report,
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()