from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from transformers import AutoModel, AutoTokenizer

from paths import DATA_DIR, OUT_DIR
from run_text_extraction_baselines import expected_calibration_error


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TARGET = "severity"
RANDOM_STATE = 42


def encode_texts(texts: pd.Series, tokenizer, model, batch_size: int = 128, max_length: int = 96) -> np.ndarray:
    model.eval()
    vectors: list[np.ndarray] = []
    values = texts.astype(str).tolist()
    device = next(model.parameters()).device
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            batch_texts = values[start : start + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            output = model(**encoded)
            token_embeddings = output.last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
            pooled = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            vectors.append(pooled.cpu().numpy().astype(np.float32))
            print(f"Encoded {min(start + batch_size, len(values)):,}/{len(values):,}", flush=True)
    return np.vstack(vectors)


def score(split_name: str, y_true: np.ndarray, y_pred: np.ndarray, probabilities: np.ndarray) -> dict[str, object]:
    precision, recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    confidence = probabilities.max(axis=1)
    return {
        "target": TARGET,
        "model": "minilm_embedding_logreg",
        "encoder": MODEL_NAME,
        "eval_split": split_name,
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "mean_confidence": float(np.mean(confidence)),
        "ece_10bin": expected_calibration_error(np.asarray(y_true), np.asarray(y_pred), confidence),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    crew = pd.read_csv(DATA_DIR / "crew_reports.csv")
    split = pd.read_csv(DATA_DIR / "split.csv")[["flight_id", "split"]]
    gold = pd.read_csv(DATA_DIR / "gold_standard_labels.csv")

    crew = crew.merge(split, on="flight_id", how="left")
    gold_eval = pd.DataFrame(
        {
            "report_text": gold["report_text"],
            TARGET: gold["target_severity"],
        }
    )

    train = crew[crew["split"] == "train"].copy()
    val = crew[crew["split"] == "val"].copy()
    test = crew[crew["split"] == "test"].copy()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)

    train_x = encode_texts(train["report_text"], tokenizer, model)
    val_x = encode_texts(val["report_text"], tokenizer, model)
    test_x = encode_texts(test["report_text"], tokenizer, model)
    gold_x = encode_texts(gold_eval["report_text"], tokenizer, model)

    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    clf.fit(train_x, train[TARGET])

    rows = []
    pred_frames = []
    for split_name, x, frame in [
        ("validation", val_x, val),
        ("test", test_x, test),
        ("gold_subset", gold_x, gold_eval),
    ]:
        pred = clf.predict(x)
        probabilities = clf.predict_proba(x)
        rows.append(score(split_name, frame[TARGET].to_numpy(), pred, probabilities))
        pred_frame = pd.DataFrame(
            {
                "eval_split": split_name,
                "row_index": np.arange(len(frame)),
                "report_text": frame["report_text"].to_numpy(),
                "true_severity": frame[TARGET].to_numpy(),
                "predicted_severity": pred,
                "prediction_confidence": probabilities.max(axis=1),
            }
        )
        pred_frames.append(pred_frame)

    results = pd.DataFrame(rows)
    results.to_csv(OUT_DIR / "phase4_transformer_severity_results.csv", index=False)
    pd.concat(pred_frames, ignore_index=True).to_csv(
        OUT_DIR / "phase4_transformer_severity_predictions.csv",
        index=False,
    )
    metadata = {
        "target": TARGET,
        "model": "MiniLM mean-pooled embeddings plus balanced logistic regression",
        "encoder": MODEL_NAME,
        "training_rows": int(len(train)),
        "validation_rows": int(len(val)),
        "test_rows": int(len(test)),
        "gold_rows": int(len(gold_eval)),
        "purpose": "Lightweight pre-trained transformer encoder baseline for the only non-deterministic extraction target.",
    }
    (OUT_DIR / "phase4_transformer_severity_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(results.to_string(index=False))
    print("Wrote:", OUT_DIR / "phase4_transformer_severity_results.csv")


if __name__ == "__main__":
    main()
