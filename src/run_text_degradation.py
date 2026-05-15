from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from paths import DATA_DIR, OUT_DIR
from run_text_extraction_baselines import make_text_model


ALPHABET = np.array(list("abcdefghijklmnopqrstuvwxyz"))


def corrupt_text(text: str, rate: float, rng: np.random.Generator) -> str:
    if rate <= 0:
        return str(text)
    chars = list(str(text))
    for i, ch in enumerate(chars):
        if not ch.isalpha():
            continue
        if rng.random() < rate:
            mode = rng.choice(["delete", "substitute", "swap"])
            if mode == "delete":
                chars[i] = ""
            elif mode == "substitute":
                chars[i] = str(rng.choice(ALPHABET))
            elif mode == "swap" and i + 1 < len(chars):
                chars[i], chars[i + 1] = chars[i + 1], chars[i]
    return "".join(chars)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    crew = pd.read_csv(DATA_DIR / "crew_reports.csv")
    split = pd.read_csv(DATA_DIR / "split.csv")[["flight_id", "split"]]
    crew = crew.merge(split, on="flight_id", how="left")

    train = crew[crew["split"] == "train"].copy()
    test = crew[crew["split"] == "test"].copy()

    rates = [0.0, 0.03, 0.06, 0.10, 0.15, 0.20]
    targets = ["component_code", "severity", "location_hint"]

    rows: list[dict[str, object]] = []
    rng = np.random.default_rng(42)
    corrupted_cache: dict[float, pd.Series] = {}
    for rate in rates:
        if rate == 0.0:
            corrupted_cache[rate] = test["report_text"].copy()
        else:
            # Deterministic per-rate corruption while preserving identical sample order.
            rate_rng = np.random.default_rng(int(rate * 10_000) + 42)
            corrupted_cache[rate] = test["report_text"].map(lambda text: corrupt_text(text, rate, rate_rng))

    for target in targets:
        model = make_text_model()
        model.fit(train["report_text"], train[target])
        for rate in rates:
            pred = model.predict(corrupted_cache[rate])
            rows.append(
                {
                    "target": target,
                    "corruption_rate": rate,
                    "n": int(len(test)),
                    "accuracy": float(accuracy_score(test[target], pred)),
                    "macro_f1": float(f1_score(test[target], pred, average="macro", zero_division=0)),
                    "weighted_f1": float(f1_score(test[target], pred, average="weighted", zero_division=0)),
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "phase4_text_degradation.csv", index=False)

    print(df.to_string(index=False))
    print("Wrote:", OUT_DIR / "phase4_text_degradation.csv")


if __name__ == "__main__":
    main()
