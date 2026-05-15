from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from paths import DATA_DIR, OUT_DIR
from run_text_extraction_baselines import make_text_model


SEVERITY_MAP = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}
COMPONENT_CODES = ["25-SEAT", "25-PANEL", "33-LIGHT", "38-GALLEY", "38-WATER"]
LOCATION_CODES = ["cabin_mid", "door_area", "cabin_rear"]


def one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def optimize_threshold(y_score_val: np.ndarray, y_binary_val: np.ndarray) -> tuple[float, float]:
    thresholds = np.linspace(float(y_score_val.min()), float(y_score_val.max()), 200)
    best_t = float(thresholds[0])
    best_f1 = -1.0
    for threshold in thresholds:
        pred = (y_score_val >= threshold).astype(int)
        score = f1_score(y_binary_val, pred, zero_division=0)
        if score > best_f1:
            best_t = float(threshold)
            best_f1 = float(score)
    return best_t, best_f1


def decision_metrics(y_score: np.ndarray, y_binary: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (y_score >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_binary, pred)),
        "precision": float(precision_score(y_binary, pred, zero_division=0)),
        "recall": float(recall_score(y_binary, pred, zero_division=0)),
        "f1": float(f1_score(y_binary, pred, zero_division=0)),
    }


def aggregate_report_features(reports: pd.DataFrame, prefix: str) -> pd.DataFrame:
    frame = reports.copy()
    frame[f"{prefix}_severity_num"] = frame["severity"].map(SEVERITY_MAP).fillna(0)
    base = frame.groupby("flight_id").agg(
        **{
            f"{prefix}_severity_max": (f"{prefix}_severity_num", "max"),
            f"{prefix}_severity_mean": (f"{prefix}_severity_num", "mean"),
        }
    )

    comp = pd.crosstab(frame["flight_id"], frame["component_code"])
    comp = comp.reindex(columns=COMPONENT_CODES, fill_value=0)
    comp.columns = [f"{prefix}_component_count_{c}" for c in comp.columns]

    loc = pd.crosstab(frame["flight_id"], frame["location_hint"])
    loc = loc.reindex(columns=LOCATION_CODES, fill_value=0)
    loc.columns = [f"{prefix}_location_count_{c}" for c in loc.columns]

    return base.join([comp, loc], how="outer").fillna(0).reset_index()


def build_extracted_reports(crew: pd.DataFrame, split: pd.DataFrame) -> pd.DataFrame:
    crew_split = crew.merge(split[["flight_id", "split"]], on="flight_id", how="left")
    train = crew_split[crew_split["split"] == "train"]
    extracted = crew_split[["flight_id", "report_text"]].copy()

    for target in ["component_code", "severity", "location_hint"]:
        model = make_text_model()
        model.fit(train["report_text"], train[target])
        extracted[target] = model.predict(crew_split["report_text"])

    return extracted


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    flights = pd.read_csv(DATA_DIR / "flights.csv")
    flights["departure"] = pd.to_datetime(flights["departure"], errors="coerce")
    labels = pd.read_csv(DATA_DIR / "maintenance_labels.csv")
    split = pd.read_csv(DATA_DIR / "split.csv")[["flight_id", "split"]]
    crew = pd.read_csv(DATA_DIR / "crew_reports.csv")

    extracted_reports = build_extracted_reports(crew, split)

    true_report_features = aggregate_report_features(
        crew[["flight_id", "component_code", "severity", "location_hint"]],
        "true",
    )
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
    true_text_features = [c for c in base.columns if c.startswith("true_")]
    extracted_text_features = [c for c in base.columns if c.startswith("extracted_")]

    feature_sets = {
        "true_reports_only": true_text_features,
        "extracted_reports_only": extracted_text_features,
        "usage_only": usage_features,
        "usage_plus_true_reports": usage_features + true_text_features,
        "usage_plus_extracted_reports": usage_features + extracted_text_features,
    }

    train = base[base["split"] == "train"].copy()
    val = base[base["split"] == "val"].copy()
    test = base[base["split"] == "test"].copy()

    models = {
        "mean_baseline": DummyRegressor(strategy="mean"),
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(n_estimators=250, min_samples_leaf=3, random_state=42, n_jobs=-1),
        "hist_gradient_boosting": HistGradientBoostingRegressor(max_iter=250, learning_rate=0.05, random_state=42),
    }

    reg_rows: list[dict[str, object]] = []
    decision_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []

    for feature_set_name, features in feature_sets.items():
        cat_cols = [c for c in features if base[c].dtype == "object"]
        num_cols = [c for c in features if c not in cat_cols]
        preprocess = ColumnTransformer(
            transformers=[
                ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_cols),
                ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", one_hot_encoder())]), cat_cols),
            ]
        )

        for model_name, model in models.items():
            pipe = Pipeline([("preprocess", preprocess), ("model", model)])
            pipe.fit(train[features], train["hazard_score"])
            val_pred = pipe.predict(val[features])
            test_pred = pipe.predict(test[features])

            for split_name, frame, pred in [("validation", val, val_pred), ("test", test, test_pred)]:
                reg_rows.append(
                    {
                        "task": "maintenance_risk_index_regression",
                        "feature_set": feature_set_name,
                        "model": model_name,
                        "eval_split": split_name,
                        "n": int(len(frame)),
                        "mae": float(mean_absolute_error(frame["hazard_score"], pred)),
                        "rmse": rmse(frame["hazard_score"], pred),
                        "r2": float(r2_score(frame["hazard_score"], pred)),
                    }
                )

            threshold, val_f1 = optimize_threshold(val_pred, val["maintenance_needed"].to_numpy())
            dec = decision_metrics(test_pred, test["maintenance_needed"].to_numpy(), threshold)
            decision_rows.append(
                {
                    "task": "maintenance_needed_from_predicted_risk_index",
                    "feature_set": feature_set_name,
                    "model": model_name,
                    "threshold_selected_on_validation": threshold,
                    "validation_f1_at_threshold": val_f1,
                    **dec,
                }
            )
            if model_name == "hist_gradient_boosting":
                pred_frame = test[["flight_id", "hazard_score", "maintenance_needed"]].copy()
                pred_frame["feature_set"] = feature_set_name
                pred_frame["model"] = model_name
                pred_frame["predicted_risk"] = test_pred
                pred_frame["threshold_selected_on_validation"] = threshold
                pred_frame["predicted_maintenance_needed"] = (test_pred >= threshold).astype(int)
                prediction_rows.append(pred_frame)

    reg_df = pd.DataFrame(reg_rows)
    decision_df = pd.DataFrame(decision_rows)
    reg_df.to_csv(OUT_DIR / "phase4_maintenance_risk_regression.csv", index=False)
    decision_df.to_csv(OUT_DIR / "phase4_maintenance_decision_thresholds.csv", index=False)
    pd.concat(prediction_rows, ignore_index=True).to_csv(OUT_DIR / "phase4_maintenance_hgb_test_predictions.csv", index=False)

    metadata = {
        "target": "hazard_score treated as synthetic maintenance risk index",
        "binary_decision": "maintenance_needed is used only for threshold-based decision evaluation; it is not directly trained as a classifier.",
        "excluded_features": [
            "tail_id",
            "raw departure timestamp",
            "pred_time_to_failure_hours",
            "maintenance_needed",
            "latent_*",
            "inventory/event features for main manuscript",
        ],
        "feature_sets": feature_sets,
        "split_sizes": {"train": int(len(train)), "validation": int(len(val)), "test": int(len(test))},
    }
    (OUT_DIR / "phase4_maintenance_benchmark_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Regression")
    print(reg_df.sort_values(["eval_split", "feature_set", "r2"], ascending=[True, True, False]).to_string(index=False))
    print("\nDecision thresholds")
    print(decision_df.sort_values(["feature_set", "f1"], ascending=[True, False]).to_string(index=False))
    print("Wrote maintenance benchmark outputs to:", OUT_DIR)


if __name__ == "__main__":
    main()
