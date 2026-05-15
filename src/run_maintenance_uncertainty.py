from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, r2_score

from paths import OUT_DIR


def ci(values: list[float]) -> tuple[float, float]:
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def bootstrap_metrics(frame: pd.DataFrame, n_boot: int = 2000, seed: int = 42) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    y_risk = frame["hazard_score"].to_numpy()
    pred_risk = frame["predicted_risk"].to_numpy()
    y_binary = frame["maintenance_needed"].to_numpy()
    pred_binary = frame["predicted_maintenance_needed"].to_numpy()
    n = len(frame)
    r2_values: list[float] = []
    f1_values: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        r2_values.append(float(r2_score(y_risk[idx], pred_risk[idx])))
        f1_values.append(float(f1_score(y_binary[idx], pred_binary[idx], zero_division=0)))
    r2_lo, r2_hi = ci(r2_values)
    f1_lo, f1_hi = ci(f1_values)
    return {
        "r2": float(r2_score(y_risk, pred_risk)),
        "r2_ci_low": r2_lo,
        "r2_ci_high": r2_hi,
        "f1": float(f1_score(y_binary, pred_binary, zero_division=0)),
        "f1_ci_low": f1_lo,
        "f1_ci_high": f1_hi,
    }


def bootstrap_paired_difference(
    base: pd.DataFrame,
    comparison: pd.DataFrame,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict[str, float]:
    base = base.sort_values("flight_id").reset_index(drop=True)
    comparison = comparison.sort_values("flight_id").reset_index(drop=True)
    if not base["flight_id"].equals(comparison["flight_id"]):
        raise ValueError("Feature-set predictions are not aligned by flight_id.")

    rng = np.random.default_rng(seed)
    y_risk = base["hazard_score"].to_numpy()
    y_binary = base["maintenance_needed"].to_numpy()
    base_risk = base["predicted_risk"].to_numpy()
    comp_risk = comparison["predicted_risk"].to_numpy()
    base_binary = base["predicted_maintenance_needed"].to_numpy()
    comp_binary = comparison["predicted_maintenance_needed"].to_numpy()
    n = len(base)

    r2_diff_values: list[float] = []
    f1_diff_values: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        r2_diff_values.append(float(r2_score(y_risk[idx], comp_risk[idx]) - r2_score(y_risk[idx], base_risk[idx])))
        f1_diff_values.append(
            float(
                f1_score(y_binary[idx], comp_binary[idx], zero_division=0)
                - f1_score(y_binary[idx], base_binary[idx], zero_division=0)
            )
        )
    r2_lo, r2_hi = ci(r2_diff_values)
    f1_lo, f1_hi = ci(f1_diff_values)
    return {
        "r2_difference_vs_usage_only": float(r2_score(y_risk, comp_risk) - r2_score(y_risk, base_risk)),
        "r2_difference_ci_low": r2_lo,
        "r2_difference_ci_high": r2_hi,
        "f1_difference_vs_usage_only": float(
            f1_score(y_binary, comp_binary, zero_division=0) - f1_score(y_binary, base_binary, zero_division=0)
        ),
        "f1_difference_ci_low": f1_lo,
        "f1_difference_ci_high": f1_hi,
    }


def main() -> None:
    predictions = pd.read_csv(OUT_DIR / "phase4_maintenance_hgb_test_predictions.csv")

    metric_rows = []
    for feature_set, frame in predictions.groupby("feature_set"):
        metric_rows.append({"feature_set": feature_set, **bootstrap_metrics(frame)})
    pd.DataFrame(metric_rows).to_csv(OUT_DIR / "phase4_maintenance_uncertainty.csv", index=False)

    base = predictions[predictions["feature_set"] == "usage_only"].copy()
    diff_rows = []
    for feature_set, frame in predictions.groupby("feature_set"):
        if feature_set == "usage_only":
            continue
        diff_rows.append(
            {
                "baseline_feature_set": "usage_only",
                "comparison_feature_set": feature_set,
                **bootstrap_paired_difference(base, frame),
            }
        )
    pd.DataFrame(diff_rows).to_csv(OUT_DIR / "phase4_maintenance_paired_differences.csv", index=False)

    print("Wrote uncertainty outputs to:", OUT_DIR)


if __name__ == "__main__":
    main()
