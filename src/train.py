from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_recall_curve, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from data import DatasetConfig, load_raw_csv, prepare_dataset
from features import build_preprocessor, infer_feature_types
from metrics import BusinessConfig, best_threshold_by_profit, classification_metrics, expected_profit


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def time_split(df, time_col, test_size):
    df_sorted = df.sort_values(time_col)
    cutoff = int(np.floor((1 - test_size) * len(df_sorted)))
    train_df = df_sorted.iloc[:cutoff].copy()
    test_df = df_sorted.iloc[cutoff:].copy()
    return train_df, test_df


def main():
    cfg = load_config("config.yaml")
    seed = cfg["project"]["seed"]
    data_path = cfg["paths"]["raw_data"]
    model_dir = Path(cfg["paths"]["model_dir"])
    report_dir = Path(cfg["paths"]["report_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    schema = cfg["schema"]
    dataset_cfg = DatasetConfig(
        target=schema["target"],
        positive_label=schema["positive_label"],
        id_cols=schema.get("id_cols", []),
        optional_time_col=schema.get("optional_time_col"),
    )

    print("Step 1/5: Load data")
    df = load_raw_csv(data_path)
    df = prepare_dataset(df, dataset_cfg)

    target = dataset_cfg.target
    positive_label = dataset_cfg.positive_label
    if df[target].isna().any():
        before = len(df)
        df = df.dropna(subset=[target]).copy()
        print(f"Warning: dropped {before - len(df)} rows with missing target.")

    target_series = df[target]
    unique_vals = set(target_series.dropna().unique().tolist())
    if positive_label not in unique_vals:
        if isinstance(positive_label, bool):
            mapped = "Yes" if positive_label else "No"
            if mapped in unique_vals:
                print(f"Warning: positive_label parsed as bool, using '{mapped}' instead.")
                positive_label = mapped
        elif str(positive_label) in unique_vals:
            print("Warning: positive_label type mismatch, using string value instead.")
            positive_label = str(positive_label)
        else:
            coerced = pd.to_numeric(pd.Series([positive_label]), errors="coerce").iloc[0]
            if pd.notna(coerced) and coerced in unique_vals:
                print("Warning: positive_label type mismatch, using numeric value instead.")
                positive_label = coerced
            else:
                raise ValueError(
                    f"positive_label '{positive_label}' not found in target values: {sorted(unique_vals)}"
                )

    y = (target_series == positive_label).astype(int)
    print(
        "Target distribution:",
        y.value_counts(dropna=False).to_dict()
        if hasattr(y, "value_counts")
        else dict(zip(*np.unique(y, return_counts=True))),
    )
    X = df.drop(columns=[target])

    print("Step 2/5: Split data")
    time_col = dataset_cfg.optional_time_col
    used_time_split = False
    if time_col and time_col in X.columns:
        used_time_split = True
        train_df = X.copy()
        train_df[target] = y
        train_df, test_df = time_split(train_df, time_col, cfg["split"]["test_size"])
        X_train = train_df.drop(columns=[target])
        y_train = train_df[target].values
        X_test = test_df.drop(columns=[target])
        y_test = test_df[target].values
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=cfg["split"]["test_size"],
            stratify=y if cfg["split"].get("stratify", True) else None,
            random_state=seed,
        )

    if used_time_split and time_col in X_train.columns:
        X_train = X_train.drop(columns=[time_col])
        X_test = X_test.drop(columns=[time_col])

    unique_train = set(np.unique(y_train))
    if len(unique_train) < 2:
        if used_time_split:
            print("Warning: time split has only one class; falling back to stratified split.")
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=cfg["split"]["test_size"],
                stratify=y if cfg["split"].get("stratify", True) else None,
                random_state=seed,
            )
        else:
            raise ValueError("Train split has only one class. Check target labels.")

    print(
        "Train distribution:",
        {int(k): int(v) for k, v in zip(*np.unique(y_train, return_counts=True))},
    )
    print(
        "Test distribution:",
        {int(k): int(v) for k, v in zip(*np.unique(y_test, return_counts=True))},
    )
    if len(np.unique(y_test)) < 2:
        print("Warning: test split has only one class; some metrics will be undefined.")

    print("Step 3/5: Build features")
    feature_sets = infer_feature_types(X_train, target=target)
    preprocessor = build_preprocessor(feature_sets)

    print("Step 4/5: Train model")
    model = LogisticRegression(
        C=cfg["model"]["params"].get("C", 1.0),
        max_iter=cfg["model"]["params"].get("max_iter", 1000),
        solver=cfg["model"]["params"].get("solver", "lbfgs"),
        class_weight="balanced",
    )

    clf = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
    clf.fit(X_train, y_train)

    calibrated = None
    calib_cfg = cfg.get("calibration", {})
    if calib_cfg.get("enabled", False):
        print("Step 4.1/5: Calibrate probabilities")
        calibrated = CalibratedClassifierCV(
            clf,
            method=calib_cfg.get("method", "sigmoid"),
            cv=calib_cfg.get("cv", 3),
        )
        calibrated.fit(X_train, y_train)

    print("Step 5/5: Evaluate and save")
    model_for_inference = calibrated if calibrated is not None else clf
    y_prob = model_for_inference.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = classification_metrics(y_test, y_prob, cfg["metrics"]["k_list"])
    business_cfg = BusinessConfig(
        retention_cost=cfg["business"]["retention_cost"],
        saved_revenue=cfg["business"]["saved_revenue"],
    )
    threshold_grid = cfg["metrics"].get("threshold_grid")
    best_t, best_profit = best_threshold_by_profit(
        y_test, y_prob, business_cfg, thresholds=threshold_grid
    )
    profit_curve = {
        str(t): expected_profit(y_test, y_prob, business_cfg, t)
        for t in (threshold_grid or [])
    }

    report = {
        "metrics": metrics,
        "best_threshold": best_t,
        "best_profit": best_profit,
        "profit_curve": profit_curve,
    }

    with open(report_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    with open(report_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(classification_report(y_test, y_pred))

    try:
        import matplotlib.pyplot as plt

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.figure()
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.tight_layout()
        plt.savefig(report_dir / "roc_curve.png", dpi=150)
        plt.close()

        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        plt.figure()
        plt.plot(recall, precision)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("PR Curve")
        plt.tight_layout()
        plt.savefig(report_dir / "pr_curve.png", dpi=150)
        plt.close()

        if profit_curve:
            thresholds = [float(t) for t in profit_curve.keys()]
            profits = [profit_curve[str(t)] for t in thresholds]
            plt.figure()
            plt.plot(thresholds, profits)
            plt.xlabel("Threshold")
            plt.ylabel("Expected Profit")
            plt.title("Profit Curve")
            plt.tight_layout()
            plt.savefig(report_dir / "profit_curve.png", dpi=150)
            plt.close()
    except Exception:
        pass

    model_path = model_dir / "logreg_pipeline.joblib"
    try:
        import joblib
        joblib.dump(model_for_inference, model_path)
    except Exception:
        pass
