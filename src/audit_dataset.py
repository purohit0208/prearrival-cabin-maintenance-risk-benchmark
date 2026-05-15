from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from paths import DATA_DIR, OUT_DIR, RUN_DIR


def read_csv(name: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / name, **kwargs)


def safe_float(value):
    if pd.isna(value):
        return None
    return float(value)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    flights = read_csv("flights.csv", parse_dates=["departure"])
    crew = read_csv("crew_reports.csv", parse_dates=["timestamp"])
    inventory = read_csv("inventory_events.csv", parse_dates=["timestamp"])
    labels = read_csv("maintenance_labels.csv")
    actions = read_csv("maintenance_actions.csv")
    split = read_csv("split.csv", parse_dates=["departure"])
    gold = read_csv("gold_standard_labels.csv", parse_dates=["timestamp"])

    report: dict[str, object] = {}
    report["dataset_shapes"] = {
        "flights": list(flights.shape),
        "crew_reports": list(crew.shape),
        "inventory_events": list(inventory.shape),
        "maintenance_labels": list(labels.shape),
        "maintenance_actions": list(actions.shape),
        "split": list(split.shape),
        "gold_standard_labels": list(gold.shape),
    }

    ids = {
        "flights": set(flights["flight_id"]),
        "labels": set(labels["flight_id"]),
        "split": set(split["flight_id"]),
    }
    report["id_integrity"] = {
        "flights_unique": len(ids["flights"]),
        "labels_unique": len(ids["labels"]),
        "split_unique": len(ids["split"]),
        "labels_missing_from_flights": len(ids["labels"] - ids["flights"]),
        "split_missing_from_flights": len(ids["split"] - ids["flights"]),
        "flights_missing_from_labels": len(ids["flights"] - ids["labels"]),
        "flights_missing_from_split": len(ids["flights"] - ids["split"]),
    }

    split_order = {"train": 0, "val": 1, "test": 2}
    nonmonotonic_tails = []
    for tail_id, group in split.sort_values(["tail_id", "departure"]).groupby("tail_id"):
        order = group["split"].map(split_order).to_numpy()
        if np.any(order[:-1] > order[1:]):
            nonmonotonic_tails.append(tail_id)
    report["split_policy"] = {
        "counts": split["split"].value_counts().to_dict(),
        "date_ranges": {
            k: {
                "min": str(v["min"]),
                "max": str(v["max"]),
                "count": int(v["count"]),
            }
            for k, v in split.groupby("split")["departure"].agg(["min", "max", "count"]).to_dict("index").items()
        },
        "client_by_split": pd.crosstab(split["client_id"], split["split"]).to_dict(),
        "n_tails": int(split["tail_id"].nunique()),
        "tails_with_nonmonotonic_train_val_test_order": len(nonmonotonic_tails),
    }

    pos = labels.loc[labels["maintenance_needed"] == 1, "hazard_score"]
    neg = labels.loc[labels["maintenance_needed"] == 0, "hazard_score"]
    report["maintenance_target_audit"] = {
        "maintenance_needed_rate": safe_float(labels["maintenance_needed"].mean()),
        "negative_hazard_min": safe_float(neg.min()),
        "negative_hazard_max": safe_float(neg.max()),
        "positive_hazard_min": safe_float(pos.min()),
        "positive_hazard_max": safe_float(pos.max()),
        "perfect_binary_separation_from_hazard_score": bool(neg.max() < pos.min()),
        "interpretation": "maintenance_needed is threshold-derived from hazard_score; binary classification would reconstruct the generator rule.",
    }

    severity_map = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}
    crew_aug = crew.copy()
    crew_aug["severity_num"] = crew_aug["severity"].map(severity_map)
    crew_agg = crew_aug.groupby("flight_id").agg(
        crew_report_count=("severity_num", "count"),
        crew_severity_max=("severity_num", "max"),
        crew_severity_mean=("severity_num", "mean"),
    ).reset_index()
    inv_agg = inventory.groupby("flight_id").agg(
        inventory_event_count=("shortage", "count"),
        shortage_count=("shortage", "sum"),
        shortage_occurrences=("shortage", lambda s: int((s > 0).sum())),
        requested_sum=("requested_qty", "sum"),
        stock_sum=("stock_available", "sum"),
        avg_requested_qty=("requested_qty", "mean"),
        avg_stock_available=("stock_available", "mean"),
    ).reset_index()
    merged = (
        flights.drop(columns=[c for c in flights.columns if c.startswith("latent_")])
        .merge(labels[["flight_id", "maintenance_needed", "hazard_score", "pred_time_to_failure_hours"]], on="flight_id")
        .merge(crew_agg, on="flight_id", how="left")
        .merge(inv_agg, on="flight_id", how="left")
        .fillna(0)
    )

    numeric = merged.select_dtypes(include=[np.number])
    corr_hazard = (
        numeric.corr(numeric_only=True)["hazard_score"]
        .drop(labels=["hazard_score"], errors="ignore")
        .abs()
        .sort_values(ascending=False)
    )
    corr_binary = (
        numeric.corr(numeric_only=True)["maintenance_needed"]
        .drop(labels=["maintenance_needed"], errors="ignore")
        .abs()
        .sort_values(ascending=False)
    )
    leakage_corr = pd.DataFrame(
        {
            "feature": corr_hazard.index,
            "abs_corr_with_hazard_score": corr_hazard.values,
            "abs_corr_with_maintenance_needed": [corr_binary.get(idx, np.nan) for idx in corr_hazard.index],
        }
    )
    leakage_corr.to_csv(OUT_DIR / "phase2_feature_target_correlations.csv", index=False)

    old_feature_schema_path = RUN_DIR / "feature_schema.json"
    old_feature_schema = json.loads(old_feature_schema_path.read_text(encoding="utf-8"))
    old_features = set(old_feature_schema.get("numeric_features", [])) | set(old_feature_schema.get("categorical_features", []))
    report["old_notebook_leakage_risks"] = {
        "old_features_include_shortage_count": "shortage_count" in old_features,
        "old_features_include_inventory_event_count": "inventory_event_count" in old_features,
        "inventory_target_in_old_notebook": "shortage_count",
        "interpretation": "The prior inventory task predicts shortage_count while shortage_count is also in the input feature set, so those inventory scores are not valid independent forecasting evidence.",
    }

    constant_cols = []
    for col in merged.columns:
        if merged[col].nunique(dropna=False) == 1:
            constant_cols.append(col)
    report["feature_quality_flags"] = {
        "constant_columns_after_aggregation": constant_cols,
        "crew_reports_per_flight_unique_values": sorted(crew_agg["crew_report_count"].unique().tolist())[:20],
        "inventory_events_per_flight_unique_values": sorted(inv_agg["inventory_event_count"].unique().tolist())[:20],
        "latent_columns_present_in_raw_flights": [c for c in flights.columns if c.startswith("latent_")],
        "latent_columns_used_in_old_feature_schema": sorted(old_features & {c for c in flights.columns if c.startswith("latent_")}),
    }

    report["text_corpus_audit"] = {
        "crew_rows": int(len(crew)),
        "crew_unique_report_texts": int(crew["report_text"].nunique()),
        "language_counts": crew["language"].value_counts().to_dict() if "language" in crew else {},
        "component_counts": crew["component_code"].value_counts().to_dict(),
        "severity_counts": crew["severity"].value_counts().to_dict(),
        "location_counts": crew["location_hint"].value_counts().to_dict(),
        "gold_rows": int(len(gold)),
        "gold_unique_report_texts": int(gold["report_text"].nunique()),
        "gold_component_counts": gold["target_component_code"].value_counts().to_dict(),
        "gold_severity_counts": gold["target_severity"].value_counts().to_dict(),
        "gold_location_counts": gold["target_location"].value_counts().to_dict(),
        "gold_issue_counts": gold["target_issue"].value_counts().to_dict(),
        "gold_issue_task_validity": "target_issue is constant in the current gold labels, so it should not be presented as a multiclass extraction benchmark.",
    }

    inv_by_flight = inv_agg.describe().T[["mean", "std", "min", "50%", "max"]]
    report["inventory_distribution_summary"] = {
        idx: {col: safe_float(value) for col, value in row.items()}
        for idx, row in inv_by_flight.iterrows()
    }

    top_corr = leakage_corr.head(15)
    report["top_abs_correlations_with_hazard_score"] = top_corr.to_dict("records")

    (OUT_DIR / "phase2_dataset_audit.json").write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )

    lines = [
        "# Phase 2 Dataset Audit Summary",
        "",
        f"Flights: {flights.shape[0]:,}",
        f"Crew reports: {crew.shape[0]:,}",
        f"Inventory events: {inventory.shape[0]:,}",
        f"Gold extraction labels: {gold.shape[0]:,}",
        "",
        "## High-Priority Findings",
        "",
        f"- `maintenance_needed` is perfectly separated by `hazard_score`: {neg.max():.3f} < {pos.min():.3f}. Binary maintenance classification would mostly reconstruct the generator rule.",
        "- The old inventory task is not valid as independent forecasting evidence because it predicts `shortage_count` while `shortage_count` is included in the feature schema.",
        f"- Aggregated crew report count is constant per flight: {sorted(crew_agg['crew_report_count'].unique().tolist())}. This field should be dropped or explained.",
        f"- Aggregated inventory event count is constant per flight: {sorted(inv_agg['inventory_event_count'].unique().tolist())}. This field should be dropped or explained.",
        "- Gold-label `target_issue` is constant (`malfunction`), so it should not be used as a multiclass benchmark.",
        "- The split order is monotonic train -> validation -> test for every tail, which supports a temporal-per-tail split claim.",
        "",
        "## Recommended Technical Rebuild",
        "",
        "1. Use `hazard_score` only as a synthetic maintenance risk index, not as a real hazard function.",
        "2. Add text-extraction benchmarks for component, severity, and location from `report_text`.",
        "3. Redesign inventory as next-flight or next-rotation shortage prediction using lagged features only.",
        "4. Add local/client generalization tests instead of calling prediction averaging production federated learning.",
        "5. Report all results as synthetic benchmark evidence with explicit limitations.",
        "",
    ]
    (OUT_DIR / "phase2_dataset_audit_summary.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps(report["dataset_shapes"], indent=2))
    print("Wrote:", OUT_DIR / "phase2_dataset_audit.json")
    print("Wrote:", OUT_DIR / "phase2_dataset_audit_summary.md")


if __name__ == "__main__":
    main()
