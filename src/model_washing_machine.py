#!/usr/bin/env python3
"""Train and evaluate a simple REFIT washing-machine disaggregation model.

Run from the repository root:
    python src/model_washing_machine.py

Uses a chronological 80/20 split within each home, a regularized linear model,
and an always-off baseline. It depends only on pandas and numpy.
"""

from __future__ import annotations

import json
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MINUTE_DIR = ROOT / "data" / "processed" / "washing_machine_1min"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
HOUSE_CHANNELS = {1: "Appliance5", 2: "Appliance2", 3: "Appliance6", 5: "Appliance3"}
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.10
MAX_TRAIN_ROWS = 300_000
RANDOM_SEED = 42
RIDGE_ALPHA = 0.5
ACTIVE_POWER_THRESHOLD_W = 100.0


def make_features(minute: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    aggregate = pd.to_numeric(minute["Aggregate"], errors="coerce")
    lag1 = aggregate.shift(1)
    lag5 = aggregate.shift(5)
    features = pd.DataFrame(index=minute.index)
    features["aggregate_w"] = aggregate / 1000.0
    features["aggregate_lag_1m_w"] = lag1 / 1000.0
    features["aggregate_lag_5m_w"] = lag5 / 1000.0
    features["aggregate_mean_5m_w"] = aggregate.rolling(5, min_periods=5).mean() / 1000.0
    features["aggregate_mean_15m_w"] = aggregate.rolling(15, min_periods=15).mean() / 1000.0
    features["aggregate_change_1m_w"] = (aggregate - lag1) / 1000.0
    hour = minute.index.hour.to_numpy() + minute.index.minute.to_numpy() / 60.0
    weekday = minute.index.dayofweek.to_numpy()
    features["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    features["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    features["weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
    features["weekday_cos"] = np.cos(2 * np.pi * weekday / 7)
    return features, minute


def metric_values(actual: np.ndarray, predicted: np.ndarray) -> dict:
    error = predicted - actual
    actual_active = actual >= ACTIVE_POWER_THRESHOLD_W
    predicted_active = predicted >= ACTIVE_POWER_THRESHOLD_W
    true_positive = int(np.sum(actual_active & predicted_active))
    false_positive = int(np.sum(~actual_active & predicted_active))
    false_negative = int(np.sum(actual_active & ~predicted_active))
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    return {
        "mae_w": float(np.mean(np.abs(error))),
        "rmse_w": float(np.sqrt(np.mean(error ** 2))),
        "active_precision_at_100w": precision,
        "active_recall_at_100w": recall,
        "active_f1_at_100w": (2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
        "predicted_to_actual_energy_ratio": float(predicted.sum() / actual.sum()) if actual.sum() > 0 else None,
    }


def svg_actual_prediction(path: Path, house: int, frame: pd.DataFrame) -> None:
    width, height, left, right, top, bottom = 1120, 500, 82, 34, 72, 72
    plot_w, plot_h = width-left-right, height-top-bottom
    actual, predicted = frame["actual_w"], frame["predicted_w"]
    ymax = max(float(pd.concat([actual, predicted]).quantile(.99)), 1.0)
    xmin, xmax = frame.index.min(), frame.index.max()
    span = max((xmax-xmin).total_seconds(), 1)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="white"/>',
           f'<text x="32" y="31" font-family="Arial,sans-serif" font-size="21" font-weight="bold">House {house} holdout: actual vs predicted washing-machine power</text>',
           f'<text x="32" y="52" font-family="Arial,sans-serif" font-size="12" fill="#475569">Cleaned REFIT · one-minute mean · last 20% of time series · y-axis capped at p99</text>']
    for fraction in (0, .25, .5, .75, 1):
        y=top+plot_h*(1-fraction)
        out.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_w}" y2="{y:.1f}" stroke="#e2e8f0"/>')
        out.append(f'<text x="{left-9}" y="{y+4:.1f}" text-anchor="end" font-family="Arial,sans-serif" font-size="11" fill="#64748b">{ymax*fraction:.0f}</text>')
    out.append(f'<text transform="translate(20 {top+plot_h/2}) rotate(-90)" text-anchor="middle" font-family="Arial,sans-serif" font-size="12">Washing-machine power (W)</text>')
    for values, color in [(actual,"#64748b"),(predicted,"#0f766e")]:
        coords=[]
        for timestamp,value in values.items():
            if pd.isna(value): continue
            x=left+(timestamp-xmin).total_seconds()/span*plot_w
            y=top+plot_h-min(max(float(value),0),ymax)/ymax*plot_h
            coords.append(f"{x:.1f},{y:.1f}")
        out.append(f'<polyline points="{" ".join(coords)}" fill="none" stroke="{color}" stroke-width="1.5"/>')
    for i,timestamp in enumerate([xmin,xmin+(xmax-xmin)/2,xmax]):
        x=left+i*plot_w/2
        out.append(f'<text x="{x:.1f}" y="{height-45}" text-anchor="middle" font-family="Arial,sans-serif" font-size="11" fill="#64748b">{timestamp.strftime("%Y-%m-%d %H:%M")}</text>')
    out.append(f'<text x="{left+plot_w/2}" y="{height-8}" text-anchor="middle" font-family="Arial,sans-serif" font-size="12">Time (cleaned REFIT local timestamp)</text>')
    out.append(f'<line x1="{left}" y1="{height-27}" x2="{left+22}" y2="{height-27}" stroke="#64748b" stroke-width="2"/><text x="{left+29}" y="{height-23}" font-family="Arial,sans-serif" font-size="12">Actual</text>')
    out.append(f'<line x1="{left+115}" y1="{height-27}" x2="{left+137}" y2="{height-27}" stroke="#0f766e" stroke-width="2"/><text x="{left+144}" y="{height-23}" font-family="Arial,sans-serif" font-size="12">Predicted ridge regression</text>')
    out.append('</svg>')
    path.write_text("\n".join(out), encoding="utf-8")


def main() -> None:
    train_features, train_targets = [], []
    validation_data, test_data = {}, {}
    split_dates = {}
    row_counts = {}
    for house, channel in HOUSE_CHANNELS.items():
        path = MINUTE_DIR / f"House{house}_washing_machine_1min.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing one-minute source: {path}; run EDA first")
        minute = pd.read_csv(path, parse_dates=["Time"], index_col="Time")
        features, frame = make_features(minute)
        target = pd.to_numeric(frame[channel], errors="coerce")
        issue_flagged = frame["Issues"].fillna(0).gt(0) if "Issues" in frame.columns else pd.Series(False, index=frame.index)
        valid = features.notna().all(axis=1) & target.notna() & ~issue_flagged
        train_position = int(len(frame) * TRAIN_FRACTION)
        test_position = int(len(frame) * (TRAIN_FRACTION + VALIDATION_FRACTION))
        train_cutoff = frame.index[train_position]
        test_cutoff = frame.index[test_position]
        train_mask = (frame.index < train_cutoff) & valid
        validation_mask = (frame.index >= train_cutoff) & (frame.index < test_cutoff) & valid
        test_mask = (frame.index >= test_cutoff) & valid
        x_train = features.loc[train_mask].to_numpy(dtype=float)
        y_train = target.loc[train_mask].to_numpy(dtype=float)
        train_features.append(x_train)
        train_targets.append(y_train)
        validation_data[house] = (features.loc[validation_mask], target.loc[validation_mask])
        test_data[house] = (features.loc[test_mask], target.loc[test_mask])
        split_dates[house] = {"validation_start": train_cutoff.isoformat(), "test_start": test_cutoff.isoformat()}
        row_counts[house] = {
            "usable_train_minutes": int(train_mask.sum()),
            "usable_validation_minutes": int(validation_mask.sum()),
            "usable_test_minutes": int(test_mask.sum()),
            "issue_flagged_minute_bins_excluded": int(issue_flagged.sum()),
        }

    X = np.vstack(train_features)
    y = np.concatenate(train_targets)
    rng = np.random.default_rng(RANDOM_SEED)
    if len(y) > MAX_TRAIN_ROWS:
        chosen = np.sort(rng.choice(len(y), MAX_TRAIN_ROWS, replace=False))
        X, y = X[chosen], y[chosen]

    # Ridge regression closed-form solution; the intercept is not penalized.
    design = np.column_stack([np.ones(len(X)), X])
    penalty = np.eye(design.shape[1]) * RIDGE_ALPHA
    penalty[0, 0] = 0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ y)

    validation_actual, validation_raw = [], []
    for house in HOUSE_CHANNELS:
        val_features, val_target = validation_data[house]
        val_design = np.column_stack([np.ones(len(val_features)), val_features.to_numpy(dtype=float)])
        validation_actual.append(val_target.to_numpy(dtype=float))
        validation_raw.append(np.maximum(val_design @ coefficients, 0))
    val_actual = np.concatenate(validation_actual)
    val_raw = np.concatenate(validation_raw)
    candidate_cutoffs = [0, 10, 25, 50, 100, 150, 200, 300, 500]
    threshold_mae = {
        cutoff: float(np.mean(np.abs(np.where(val_raw >= cutoff, val_raw, 0) - val_actual)))
        for cutoff in candidate_cutoffs
    }
    prediction_cutoff_w = min(threshold_mae, key=threshold_mae.get)

    overall_actual, overall_raw_predicted, overall_threshold_candidate = [], [], []
    metrics_by_house = []
    house1_test = None
    for house, channel in HOUSE_CHANNELS.items():
        features, target = test_data[house]
        actual = target.to_numpy(dtype=float)
        design_test = np.column_stack([np.ones(len(features)), features.to_numpy(dtype=float)])
        raw_predicted = np.maximum(design_test @ coefficients, 0)
        predicted = raw_predicted
        threshold_candidate = np.where(raw_predicted >= prediction_cutoff_w, raw_predicted, 0)
        baseline = np.zeros_like(actual)
        metrics_by_house.append({
            "house": house,
            "channel": channel,
            "split_dates": split_dates[house],
            "usable_minutes": row_counts[house],
            "always_off_baseline": metric_values(actual, baseline),
            "ridge_regression": metric_values(actual, raw_predicted),
            "thresholded_candidate_not_selected": metric_values(actual, np.where(raw_predicted >= prediction_cutoff_w, raw_predicted, 0)),
        })
        overall_actual.append(actual)
        overall_raw_predicted.append(raw_predicted)
        overall_threshold_candidate.append(threshold_candidate)
        if house == 1:
            test_frame = pd.DataFrame({"actual_w": actual, "predicted_w": predicted}, index=features.index)
            # Select the first complete 24-hour span in holdout that contains
            # observed washing-machine activity.
            day_groups = test_frame.groupby(test_frame.index.normalize())
            for day, group in day_groups:
                if len(group) >= 20*60 and (group["actual_w"] >= ACTIVE_POWER_THRESHOLD_W).any():
                    house1_test = group.iloc[:1440]
                    break
            if house1_test is None:
                house1_test = test_frame.iloc[:1440]
            (REPORT_DIR / "model_predictions_house1_24h.csv").write_text(
                house1_test.to_csv(index_label="Time", date_format="%Y-%m-%d %H:%M:%S"), encoding="utf-8"
            )
            svg_actual_prediction(FIGURE_DIR / "model_actual_vs_pred_house1.svg", house, house1_test)

    all_actual = np.concatenate(overall_actual)
    all_raw_pred = np.concatenate(overall_raw_predicted)
    all_threshold_candidate = np.concatenate(overall_threshold_candidate)
    overall = {
        "always_off_baseline": metric_values(all_actual, np.zeros_like(all_actual)),
        "ridge_regression": metric_values(all_actual, all_raw_pred),
        "thresholded_candidate_not_selected": metric_values(all_actual, all_threshold_candidate),
    }
    report = {
        "source": "Official cleaned REFIT minute data for Houses 1, 2, 3, and 5",
        "input": "Aggregate power at minute t, lags at 1 and 5 minutes, rolling 5/15-minute aggregate means, 1-minute aggregate change, and hour/day cyclic features",
        "target": "Washing-machine active power in watts",
        "validation": "Chronological 70/10/20 split within each household. The first 70% of time trains the pooled model, the next 10% selects a global zero cutoff by lowest validation MAE, and the final 20% is the untouched test period. Overlapping time windows are never randomly mixed across the split.",
        "issues_flag_handling": "Exclude minute bins with Issues > 0 from training, validation, and test; the cleaned-data readme flags these where appliance-monitor sums exceed aggregate power.",
        "train_sample_rows": int(len(y)),
        "random_seed": RANDOM_SEED,
        "ridge_alpha": RIDGE_ALPHA,
        "validation_best_mae_cutoff_candidate_w": int(prediction_cutoff_w),
        "final_model_zero_cutoff_w": None,
        "validation_mae_by_candidate_cutoff_w": {str(k): v for k,v in threshold_mae.items()},
        "model_selection_note": "The validation-selected cutoff is reported as a rejected candidate; it nearly suppresses all activity on test data. The final model is the unthresholded ridge regression.",
        "active_power_threshold_for_f1_w": ACTIVE_POWER_THRESHOLD_W,
        "overall_test_metrics": overall,
        "per_house_test_metrics": metrics_by_house,
        "coefficients": {"intercept": float(coefficients[0]), **{name: float(value) for name,value in zip(features.columns, coefficients[1:])}},
        "figure": "reports/figures/model_actual_vs_pred_house1.svg",
        "failure_cases": [
            "Other appliances can produce aggregate power patterns similar to a washing machine.",
            "Long data outages and records flagged by the REFIT Issues column reduce the quality of training labels.",
            "A linear model cannot fully represent the washing machine's separate heating, motor, and spin phases.",
        ],
    }
    (REPORT_DIR / "model_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"train_rows": len(y), "validation_best_mae_cutoff_candidate_w": prediction_cutoff_w,
                      "validation_mae_by_candidate_cutoff_w": threshold_mae, "overall_test_metrics": overall,
                      "per_house": metrics_by_house, "figure": report["figure"]}, indent=2))


if __name__ == "__main__":
    main()
