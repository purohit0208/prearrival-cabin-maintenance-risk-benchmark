from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from paths import DATA_DIR, OUT_DIR
from run_text_extraction_baselines import make_text_model


TARGETS = ["component_code", "severity", "location_hint"]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    crew = pd.read_csv(DATA_DIR / "crew_reports.csv")
    split = pd.read_csv(DATA_DIR / "split.csv")[["flight_id", "split"]]
    crew = crew.merge(split, on="flight_id", how="left")

    train = crew[crew["split"] == "train"].copy()
    test = crew[crew["split"] == "test"].copy()
    train_texts = set(train["report_text"])
    test_overlap = test[test["report_text"].isin(train_texts)].copy()
    test_no_overlap = test[~test["report_text"].isin(train_texts)].copy()

    unique_test_texts = set(test["report_text"])
    unique_overlap = unique_test_texts & train_texts

    summary = {
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_unique_report_texts": int(train["report_text"].nunique()),
        "test_unique_report_texts": int(test["report_text"].nunique()),
        "test_unique_report_texts_seen_in_train": int(len(unique_overlap)),
        "test_unique_report_text_overlap_fraction": float(len(unique_overlap) / len(unique_test_texts)),
        "test_rows_seen_in_train": int(len(test_overlap)),
        "test_row_overlap_fraction": float(len(test_overlap) / len(test)),
        "test_rows_not_seen_in_train": int(len(test_no_overlap)),
    }

    conflict_rows: list[dict[str, object]] = []
    no_overlap_rows: list[dict[str, object]] = []

    for target in TARGETS:
        conflicts = crew.groupby("report_text")[target].nunique()
        conflict_rows.append(
            {
                "target": target,
                "unique_report_texts": int(len(conflicts)),
                "unique_report_texts_with_multiple_labels": int((conflicts > 1).sum()),
                "conflict_fraction": float((conflicts > 1).mean()),
            }
        )

        model = make_text_model()
        model.fit(train["report_text"], train[target])

        for eval_name, frame in [
            ("test_all_rows", test),
            ("test_rows_seen_in_train", test_overlap),
            ("test_rows_not_seen_in_train", test_no_overlap),
        ]:
            pred = model.predict(frame["report_text"])
            no_overlap_rows.append(
                {
                    "target": target,
                    "eval_subset": eval_name,
                    "n": int(len(frame)),
                    "accuracy": float(accuracy_score(frame[target], pred)),
                    "macro_f1": float(f1_score(frame[target], pred, average="macro", zero_division=0)),
                    "weighted_f1": float(f1_score(frame[target], pred, average="weighted", zero_division=0)),
                }
            )

    conflict_df = pd.DataFrame(conflict_rows)
    no_overlap_df = pd.DataFrame(no_overlap_rows)

    conflict_df.to_csv(OUT_DIR / "phase4_text_template_label_conflicts.csv", index=False)
    no_overlap_df.to_csv(OUT_DIR / "phase4_text_no_overlap_evaluation.csv", index=False)
    (OUT_DIR / "phase4_text_template_audit.json").write_text(
        json.dumps(
            {
                "summary": summary,
                "label_conflicts": conflict_rows,
                "no_overlap_evaluation": no_overlap_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        "# Text Extraction Template Audit",
        "",
        "Purpose: quantify whether the near-perfect component/location extraction results are a valid benchmark result or an artefact of deterministic synthetic templates.",
        "",
        "## Exact Text Overlap",
        "",
        f"- Train rows: {summary['train_rows']:,}",
        f"- Test rows: {summary['test_rows']:,}",
        f"- Train unique report texts: {summary['train_unique_report_texts']:,}",
        f"- Test unique report texts: {summary['test_unique_report_texts']:,}",
        f"- Test unique report texts also seen in train: {summary['test_unique_report_texts_seen_in_train']:,} ({summary['test_unique_report_text_overlap_fraction']:.3f})",
        f"- Test rows whose exact report text was seen in train: {summary['test_rows_seen_in_train']:,} ({summary['test_row_overlap_fraction']:.3f})",
        f"- Test rows with report text not seen in train: {summary['test_rows_not_seen_in_train']:,}",
        "",
        "## Label Conflicts Per Unique Report Text",
        "",
        conflict_df.to_markdown(index=False),
        "",
        "## Character TF-IDF Model On Seen vs Unseen Test Texts",
        "",
        no_overlap_df.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "- Component and location labels are deterministic or nearly deterministic from surface text in this synthetic corpus.",
        "- Component has no label conflicts per unique report text and remains perfect even on test rows whose exact report text was not seen during training.",
        "- Location has no label conflicts per unique report text, but no-overlap test performance drops below the all-test result.",
        "- Severity has many conflicts per unique report text and is therefore the more meaningful extraction bottleneck.",
        "- The manuscript should present component and location extraction as synthetic-template sanity checks, not as evidence of difficult aviation NLP solved by the model.",
    ]
    (OUT_DIR / "phase4_text_template_audit.md").write_text("\n".join(lines), encoding="utf-8")

    print((OUT_DIR / "phase4_text_template_audit.md").resolve())


if __name__ == "__main__":
    main()
