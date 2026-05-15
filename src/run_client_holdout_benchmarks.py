from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from paths import DATA_DIR, OUT_DIR
from run_maintenance_risk_benchmarks import (
    aggregate_report_features,
    decision_metrics,
    one_hot_encoder,
    optimize_threshold,
    rmse,
)
from run_text_extraction_baselines import make_text_model


def build_extracted_reports_for_holdout(crew_split: pd.DataFrame, holdout_client: str) -> pd.DataFrame:
    train_reports = crew_split[(crew_split["split"] == "train") & (crew_split["client_id"] != holdout_client)]
    extracted = crew_split[["flight_id", "report_text"]].copy()
    for target in ["component_code", "severity", "location_hint"]:
        model = make_text_model()
        model.fit(train_reports["report_text"], train_reports[target])
        extracted[target] = model.predict(crew_split["report_text"])
    return extracted


def make_preprocess(frame: pd.DataFrame, features: list[str]) -> ColumnTransformer:
    cat_cols = [c for c in features if frame[c].dtype == "object"]
    num_cols = [c for c in features if c not in cat_cols]
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_cols),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", one_hot_encoder())]), cat_cols),
        ]
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    flights = pd.read_csv(DATA_DIR / "flights.csv")
    flights["departure"] = pd.to_datetime(flights["departure"], errors="coerce")
    labels = pd.read_csv(DATA_DIR / "maintenance_labels.csv")
    split = pd.read_csv(DATA_DIR / "split.csv")[["flight_id", "split"]]
    crew = pd.read_csv(DATA_DIR / "crew_reports.csv")
    crew_split = crew.merge(split, on="flight_id", how="left")

    true_report_features = aggregate_report_features(
        crew[["flight_id", "component_code", "severity", "location_hint"]],
        "true",
    )

    usage_features = [
        "client_id",
        "aircraft",
        "route",
        "weather",
        "departure_hour",
        "flight_hours",
        "passengers",
        "env_factor",
        "hours_since_install_seat",
        "hours_since_install_lighting",
        "hours_since_install_galley",
        "hours_since_install_panel",
        "hours_since_install_water_waste",
        "cycles_since_install_seat",
        "cycles_since_install_lighting",
        "cycles_since_install_galley",
        "cycles_since_install_panel",
        "cycles_since_install_water_waste",
    ]

    rows: list[dict[str, object]] = []
    clients = sorted(flights["client_id"].unique())

    for holdout_client in clients:
        extracted_reports = build_extracted_reports_for_holdout(crew_split, holdout_client)
        extracted_report_features = aggregate_report_features(extracted_reports, "extracted")

        base = flights.drop(columns=[c for c in flights.columns if c.startswith("latent_")]).copy()
        base["departure_hour"] = base["departure"].dt.hour
        base = base.drop(columns=["departure", "tail_id"])
        base = (
            base.merge(labels[["flight_id", "hazard_score", "maintenance_needed"]], on="flight_id")
            .merge(split, on="flight_id")
            .merge(true_report_features, on="flight_id", how="left")
            .merge(extracted_report_features, on="flight_id", how="left")
            .fillna(0)
        )

        true_text_features = [c for c in base.columns if c.startswith("true_")]
        extracted_text_features = [c for c in base.columns if c.startswith("extracted_")]
        feature_sets = {
            "true_reports_only": true_text_features,
            "extracted_reports_only": extracted_text_features,
            "usage_only": usage_features,
            "usage_plus_true_reports": usage_features + true_text_features,
            "usage_plus_extracted_reports": usage_features + extracted_text_features,
        }

        train = base[(base["split"] == "train") & (base["client_id"] != holdout_client)]
        val = base[(base["split"] == "val") & (base["client_id"] != holdout_client)]
        test = base[(base["split"] == "test") & (base["client_id"] == holdout_client)]

        for feature_set_name, features in feature_sets.items():
            pipe = Pipeline(
                [
                    ("preprocess", make_preprocess(base, features)),
                    ("model", HistGradientBoostingRegressor(max_iter=250, learning_rate=0.05, random_state=42)),
                ]
            )
            pipe.fit(train[features], train["hazard_score"])
            val_pred = pipe.predict(val[features])
            test_pred = pipe.predict(test[features])
            threshold, val_f1 = optimize_threshold(val_pred, val["maintenance_needed"].to_numpy())
            dec = decision_metrics(test_pred, test["maintenance_needed"].to_numpy(), threshold)
            rows.append(
                {
                    "holdout_client": holdout_client,
                    "feature_set": feature_set_name,
                    "train_n": int(len(train)),
                    "val_n": int(len(val)),
                    "test_n": int(len(test)),
                    "test_mae": float(mean_absolute_error(test["hazard_score"], test_pred)),
                    "test_rmse": rmse(test["hazard_score"], test_pred),
                    "test_r2": float(r2_score(test["hazard_score"], test_pred)),
                    "validation_f1_at_threshold": val_f1,
                    **{f"decision_{k}": v for k, v in dec.items()},
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "phase4_client_holdout_maintenance.csv", index=False)
    summary = (
        df.groupby("feature_set")
        .agg(
            mean_test_r2=("test_r2", "mean"),
            min_test_r2=("test_r2", "min"),
            mean_decision_f1=("decision_f1", "mean"),
            min_decision_f1=("decision_f1", "min"),
        )
        .reset_index()
    )
    summary.to_csv(OUT_DIR / "phase4_client_holdout_maintenance_summary.csv", index=False)

    metadata = {
        "protocol": "For each held-out client, text extractors and maintenance models are trained on train flights from the other clients, thresholds are selected on validation flights from other clients, and evaluation is done on held-out client test flights. Inventory features are not used in the maintenance-only benchmark.",
        "clients": clients,
    }
    (OUT_DIR / "phase4_client_holdout_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(df.to_string(index=False))
    print("\nSummary")
    print(summary.to_string(index=False))
    print("Wrote:", OUT_DIR / "phase4_client_holdout_maintenance.csv")


if __name__ == "__main__":
    main()
