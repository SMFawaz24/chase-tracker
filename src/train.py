"""Train a 2nd-innings win-probability model (XGBoost) with a time-based split.

Usage:
    python src/train.py --data data/processed/deliveries.parquet
Prints log-loss, Brier score, and the baseline; saves model, venue map, metrics and calibration plot.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss
from xgboost import XGBClassifier

from features import FEATURES, build_features

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed/deliveries.parquet")
    args = ap.parse_args()

    d, venue_map = build_features(pd.read_parquet(args.data))

    # Time-based split: last 2 seasons are the test set -> no leakage from the future
    seasons = sorted(d["season"].unique())
    test_seasons = seasons[-2:]
    train, test = d[~d["season"].isin(test_seasons)], d[d["season"].isin(test_seasons)]
    print(f"Train: {len(train):,} balls | Test: {len(test):,} balls (seasons {test_seasons})")

    model = XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8, eval_metric="logloss")
    model.fit(train[FEATURES], train["y"])
    p = model.predict_proba(test[FEATURES])[:, 1]

    base = np.full(len(test), train["y"].mean())
    metrics = {
        "train_balls": int(len(train)),
        "test_balls": int(len(test)),
        "test_seasons": test_seasons,
        "log_loss": float(log_loss(test["y"], p)),
        "log_loss_baseline": float(log_loss(test["y"], base)),
        "brier": float(brier_score_loss(test["y"], p)),
        "brier_baseline": float(brier_score_loss(test["y"], base)),
    }
    print(f"Log-loss : {metrics['log_loss']:.4f}  (baseline {metrics['log_loss_baseline']:.4f})")
    print(f"Brier    : {metrics['brier']:.4f}  (baseline {metrics['brier_baseline']:.4f})")

    frac_pos, mean_pred = calibration_curve(test["y"], p, n_bins=10)
    plt.plot(mean_pred, frac_pos, marker="o", label="model")
    plt.plot([0, 1], [0, 1], "--", label="perfect")
    plt.xlabel("Predicted win prob")
    plt.ylabel("Observed win rate")
    plt.legend()
    plt.title("Calibration (held-out seasons)")
    Path("reports").mkdir(exist_ok=True)
    plt.savefig("reports/calibration.png", dpi=150, bbox_inches="tight")

    Path("models").mkdir(exist_ok=True)
    model.save_model("models/winprob.json")
    Path("models/venues.json").write_text(json.dumps(venue_map))
    Path("models/metrics.json").write_text(json.dumps(metrics, indent=2))
    print("Saved models/winprob.json, models/venues.json, models/metrics.json, reports/calibration.png")
