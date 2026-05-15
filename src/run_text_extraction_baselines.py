from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline

from paths import DATA_DIR, OUT_DIR


COMPONENT_RULES = {
    "25-SEAT": ["seat", "armrest", "tray", "recline", "cushion"],
    "33-LIGHT": ["light", "lamp", "reading", "flicker"],
    "38-GALLEY": ["galley", "oven", "coffee", "trolley", "chiller", "heater"],
    "25-PANEL": ["panel", "door", "creak", "buzz", " r1", " r2", " l1", " l2"],
    "38-WATER": ["sink", "water", "lavatory", "leak", "moisture", "boiler"],
}

LOCATION_RULES = {
    "cabin_rear": ["aft", "rear"],
    "door_area": [" r1", " r2", " l1", " l2", "door"],
    "cabin_mid": ["fwd", "mid", "row", "reihe", "cabin"],
}

SEVERITY_RULES = {
    "Critical": ["overheat", "alarm", "critical", "smoke", "unsafe"],
    "High": ["error", "leak", "failure", "loose", "irregular", "inconsistent"],
    "Medium": ["intermittent", "fluctuat", "flicker", "misaligned", "fitment"],
    "Low": ["small", "needs fixing", "hard to adjust"],
}


def normalize_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return f" {text.strip()} "


def most_common(series: pd.Series) -> str:
    return str(series.value_counts().idxmax())


def rule_predict(texts: Iterable[str], rules: dict[str, list[str]], fallback: str) -> tuple[list[str], list[float]]:
    preds: list[str] = []
    confs: list[float] = []
    for raw in texts:
        text = normalize_text(raw)
        best_label = fallback
        best_hits = 0
        for label, patterns in rules.items():
            hits = sum(1 for pattern in patterns if pattern in text)
            if hits > best_hits:
                best_label = label
                best_hits = hits
        preds.append(best_label)
        confs.append(0.55 + 0.1 * min(best_hits, 4) if best_hits else 0.35)
    return preds, confs


def expected_calibration_error(y_true: np.ndarray, y_pred: np.ndarray, confidence: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if not np.any(mask):
            continue
        acc = np.mean(y_true[mask] == y_pred[mask])
        conf = np.mean(confidence[mask])
        ece += np.mean(mask) * abs(acc - conf)
    return float(ece)


def score_predictions(target: str, model: str, split_name: str, y_true, y_pred, confidence=None) -> dict[str, object]:
    precision, recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    result: dict[str, object] = {
        "target": target,
        "model": model,
        "eval_split": split_name,
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
    }
    if confidence is not None:
        result["mean_confidence"] = float(np.mean(confidence))
        result["ece_10bin"] = expected_calibration_error(np.asarray(y_true), np.asarray(y_pred), np.asarray(confidence))
    return result


def make_text_model() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                SGDClassifier(
                    loss="log_loss",
                    alpha=1e-5,
                    max_iter=40,
                    tol=1e-3,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def ml_predict_with_confidence(model: Pipeline, texts: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    preds = model.predict(texts)
    if hasattr(model[-1], "predict_proba"):
        probs = model.predict_proba(texts)
        conf = probs.max(axis=1)
    else:
        conf = np.ones(len(preds))
    return preds, conf


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    crew = pd.read_csv(DATA_DIR / "crew_reports.csv")
    split = pd.read_csv(DATA_DIR / "split.csv")[["flight_id", "split"]]
    gold = pd.read_csv(DATA_DIR / "gold_standard_labels.csv")

    crew = crew.merge(split, on="flight_id", how="left")
    gold = gold.merge(split, on="flight_id", how="left")

    gold_eval = pd.DataFrame(
        {
            "report_text": gold["report_text"],
            "component_code": gold["target_component_code"],
            "severity": gold["target_severity"],
            "location_hint": gold["target_location"],
        }
    )

    target_specs = {
        "component_code": COMPONENT_RULES,
        "severity": SEVERITY_RULES,
        "location_hint": LOCATION_RULES,
    }

    results: list[dict[str, object]] = []
    class_rows: list[dict[str, object]] = []

    train = crew[crew["split"] == "train"].copy()
    test = crew[crew["split"] == "test"].copy()
    val = crew[crew["split"] == "val"].copy()

    for target, rules in target_specs.items():
        fallback = most_common(train[target])

        for eval_name, eval_df in [("validation", val), ("test", test), ("gold_subset", gold_eval)]:
            rule_preds, rule_confs = rule_predict(eval_df["report_text"], rules, fallback)
            results.append(
                score_predictions(
                    target,
                    "keyword_rules",
                    eval_name,
                    eval_df[target].to_numpy(),
                    np.asarray(rule_preds),
                    np.asarray(rule_confs),
                )
            )

        model = make_text_model()
        model.fit(train["report_text"], train[target])

        for eval_name, eval_df in [("validation", val), ("test", test), ("gold_subset", gold_eval)]:
            pred, conf = ml_predict_with_confidence(model, eval_df["report_text"])
            results.append(score_predictions(target, "char_tfidf_sgd", eval_name, eval_df[target].to_numpy(), pred, conf))

        labels = sorted(train[target].unique())
        pred, _ = ml_predict_with_confidence(model, test["report_text"])
        per_class = precision_recall_fscore_support(test[target], pred, labels=labels, zero_division=0)
        for label, p, r, f1, support in zip(labels, *per_class):
            class_rows.append(
                {
                    "target": target,
                    "model": "char_tfidf_sgd",
                    "eval_split": "test",
                    "class": label,
                    "precision": float(p),
                    "recall": float(r),
                    "f1": float(f1),
                    "support": int(support),
                }
            )

    results_df = pd.DataFrame(results)
    class_df = pd.DataFrame(class_rows)

    results_df.to_csv(OUT_DIR / "phase4_text_extraction_results.csv", index=False)
    class_df.to_csv(OUT_DIR / "phase4_text_extraction_per_class_test.csv", index=False)
    (OUT_DIR / "phase4_text_extraction_results.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print(results_df.sort_values(["target", "eval_split", "model"]).to_string(index=False))
    print("Wrote:", OUT_DIR / "phase4_text_extraction_results.csv")


if __name__ == "__main__":
    main()
