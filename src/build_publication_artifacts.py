from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from paths import OUT_DIR

OUT = OUT_DIR
ARTIFACTS = OUT / "publication_artifacts"
TABLES = ARTIFACTS / "tables"
FIGURES = ARTIFACTS / "figures"


FEATURE_LABELS = {
    "true_reports_only": "True reports only",
    "extracted_reports_only": "Extracted reports only",
    "usage_only": "Usage only",
    "usage_plus_true_reports": "Usage + true reports",
    "usage_plus_extracted_reports": "Usage + extracted reports",
}

MAINTENANCE_FEATURE_SETS = list(FEATURE_LABELS.keys())

TARGET_LABELS = {
    "component_code": "Component",
    "severity": "Severity",
    "location_hint": "Location",
}


def ensure_dirs() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)


def fmt_value(value, digits: int = 3) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (int,)) or (isinstance(value, float) and float(value).is_integer() and abs(value) >= 10):
        return str(int(value))
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def to_markdown_table(df: pd.DataFrame, title: str | None = None) -> str:
    lines: list[str] = []
    if title:
        lines.extend([f"# {title}", ""])
    headers = list(df.columns)
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt_value(row[col]) for col in headers) + " |")
    lines.append("")
    return "\n".join(lines)


def save_table(df: pd.DataFrame, stem: str, title: str) -> None:
    df.to_csv(TABLES / f"{stem}.csv", index=False)
    (TABLES / f"{stem}.md").write_text(to_markdown_table(df, title), encoding="utf-8")


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIGURES / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def table_dataset_summary() -> None:
    audit = json.loads((OUT / "phase2_dataset_audit.json").read_text(encoding="utf-8"))
    shapes = audit["dataset_shapes"]
    split_counts = audit["split_policy"]["counts"]
    maintenance = audit["maintenance_target_audit"]
    text = audit["text_corpus_audit"]

    rows = [
        ("Flights", shapes["flights"][0]),
        ("Crew report rows", shapes["crew_reports"][0]),
        ("Maintenance label rows", shapes["maintenance_labels"][0]),
        ("Gold text label rows", shapes["gold_standard_labels"][0]),
        ("Train flights", split_counts.get("train", "")),
        ("Validation flights", split_counts.get("val", "")),
        ("Test flights", split_counts.get("test", "")),
        ("Unique crew report texts", text["crew_unique_report_texts"]),
        ("Maintenance-needed rate", maintenance["maintenance_needed_rate"]),
    ]
    df = pd.DataFrame(rows, columns=["Item", "Value"])
    save_table(df, "table1_dataset_summary", "Table 1. Dataset and Split Summary")


def table_text_extraction() -> None:
    df = pd.read_csv(OUT / "phase4_text_extraction_results.csv")
    df = df[(df["model"] == "char_tfidf_sgd") & (df["eval_split"].isin(["test", "gold_subset"]))]
    transformer_path = OUT / "phase4_transformer_severity_results.csv"
    if transformer_path.exists():
        transformer = pd.read_csv(transformer_path)
        transformer = transformer[transformer["eval_split"].isin(["test", "gold_subset"])]
        df = pd.concat([df, transformer], ignore_index=True, sort=False)
    rows = []
    model_labels = {
        "char_tfidf_sgd": "Char TF-IDF + SGD",
        "minilm_embedding_logreg": "MiniLM embeddings + logistic regression",
    }
    for (target, model), group in df.groupby(["target", "model"]):
        test = group[group["eval_split"] == "test"].iloc[0]
        gold = group[group["eval_split"] == "gold_subset"].iloc[0]
        rows.append(
            {
                "Target": TARGET_LABELS.get(target, target),
                "Model": model_labels.get(model, model),
                "Test accuracy": test["accuracy"],
                "Test macro-F1": test["macro_f1"],
                "Test ECE": test["ece_10bin"],
                "Gold accuracy": gold["accuracy"],
                "Gold macro-F1": gold["macro_f1"],
                "Gold ECE": gold["ece_10bin"],
            }
        )
    order = {
        ("Component", "Char TF-IDF + SGD"): 0,
        ("Severity", "Char TF-IDF + SGD"): 1,
        ("Severity", "MiniLM embeddings + logistic regression"): 2,
        ("Location", "Char TF-IDF + SGD"): 3,
    }
    out = pd.DataFrame(rows)
    out["_order"] = out.apply(lambda row: order.get((row["Target"], row["Model"]), 99), axis=1)
    out = out.sort_values("_order").drop(columns=["_order"])
    save_table(out, "table2_text_extraction", "Table 2. Text Extraction Performance")


def table_text_degradation() -> None:
    df = pd.read_csv(OUT / "phase4_text_degradation.csv")
    df["Target"] = df["target"].map(TARGET_LABELS)
    keep_rates = [0.0, 0.1, 0.2]
    df = df[df["corruption_rate"].isin(keep_rates)]
    pivot = (
        df.pivot(index="Target", columns="corruption_rate", values="macro_f1")
        .reset_index()
        .rename(columns={0.0: "Macro-F1 at 0%", 0.1: "Macro-F1 at 10%", 0.2: "Macro-F1 at 20%"})
    )
    order = {"Component": 0, "Severity": 1, "Location": 2}
    pivot = pivot.sort_values("Target", key=lambda s: s.map(order))
    save_table(pivot, "table3_text_degradation", "Table 3. Text Degradation Robustness")


def table_maintenance_regression() -> None:
    df = pd.read_csv(OUT / "phase4_maintenance_risk_regression.csv")
    df = df[(df["model"] == "hist_gradient_boosting") & (df["eval_split"] == "test")].copy()
    df = df[df["feature_set"].isin(MAINTENANCE_FEATURE_SETS)]
    df["Feature set"] = df["feature_set"].map(FEATURE_LABELS)
    out = df[["Feature set", "mae", "rmse", "r2"]].rename(
        columns={"mae": "MAE", "rmse": "RMSE", "r2": "R2"}
    )
    order = {label: i for i, label in enumerate(FEATURE_LABELS.values())}
    out = out.sort_values("Feature set", key=lambda s: s.map(order))
    save_table(out, "table4_maintenance_regression", "Table 4. Maintenance-Risk Regression Ablation")


def table_maintenance_decision() -> None:
    df = pd.read_csv(OUT / "phase4_maintenance_decision_thresholds.csv")
    df = df[df["model"] == "hist_gradient_boosting"].copy()
    df = df[df["feature_set"].isin(MAINTENANCE_FEATURE_SETS)]
    df["Feature set"] = df["feature_set"].map(FEATURE_LABELS)
    out = df[
        [
            "Feature set",
            "threshold_selected_on_validation",
            "accuracy",
            "precision",
            "recall",
            "f1",
        ]
    ].rename(
        columns={
            "threshold_selected_on_validation": "Validation-selected threshold",
            "accuracy": "Accuracy",
            "precision": "Precision",
            "recall": "Recall",
            "f1": "F1",
        }
    )
    order = {label: i for i, label in enumerate(FEATURE_LABELS.values())}
    out = out.sort_values("Feature set", key=lambda s: s.map(order))
    save_table(out, "table5_maintenance_decision", "Table 5. Thresholded Maintenance Decision")


def table_client_holdout() -> None:
    df = pd.read_csv(OUT / "phase4_client_holdout_maintenance_summary.csv")
    df = df[df["feature_set"].isin(MAINTENANCE_FEATURE_SETS)]
    df["Feature set"] = df["feature_set"].map(FEATURE_LABELS)
    out = df[
        [
            "Feature set",
            "mean_test_r2",
            "min_test_r2",
            "mean_decision_f1",
            "min_decision_f1",
        ]
    ].rename(
        columns={
            "mean_test_r2": "Mean test R2",
            "min_test_r2": "Minimum test R2",
            "mean_decision_f1": "Mean decision F1",
            "min_decision_f1": "Minimum decision F1",
        }
    )
    order = {label: i for i, label in enumerate(FEATURE_LABELS.values())}
    out = out.sort_values("Feature set", key=lambda s: s.map(order))
    save_table(out, "table6_client_holdout", "Table 6. Client-Held-Out Maintenance Generalization")


def table_maintenance_uncertainty() -> None:
    metrics = pd.read_csv(OUT / "phase4_maintenance_uncertainty.csv")
    diffs = pd.read_csv(OUT / "phase4_maintenance_paired_differences.csv")
    diff_map = diffs.set_index("comparison_feature_set").to_dict(orient="index")
    rows = []
    for _, row in metrics.iterrows():
        feature_set = row["feature_set"]
        diff = diff_map.get(feature_set, {})
        rows.append(
            {
                "Feature set": FEATURE_LABELS.get(feature_set, feature_set),
                "R2": row["r2"],
                "R2 95% CI": f"{row['r2_ci_low']:.3f} to {row['r2_ci_high']:.3f}",
                "F1": row["f1"],
                "F1 95% CI": f"{row['f1_ci_low']:.3f} to {row['f1_ci_high']:.3f}",
                "Delta R2 vs usage (95% CI)": (
                    ""
                    if not diff
                    else f"{diff['r2_difference_vs_usage_only']:.3f} ({diff['r2_difference_ci_low']:.3f} to {diff['r2_difference_ci_high']:.3f})"
                ),
                "Delta F1 vs usage (95% CI)": (
                    ""
                    if not diff
                    else f"{diff['f1_difference_vs_usage_only']:.3f} ({diff['f1_difference_ci_low']:.3f} to {diff['f1_difference_ci_high']:.3f})"
                ),
            }
        )
    out = pd.DataFrame(rows)
    order = {label: i for i, label in enumerate(FEATURE_LABELS.values())}
    out = out.sort_values("Feature set", key=lambda s: s.map(order))
    save_table(out, "table7_maintenance_uncertainty", "Table 7. Maintenance Uncertainty and Effect Sizes")


def figure_text_degradation() -> None:
    df = pd.read_csv(OUT / "phase4_text_degradation.csv")
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for target, group in df.groupby("target"):
        group = group.sort_values("corruption_rate")
        ax.plot(
            group["corruption_rate"] * 100,
            group["macro_f1"],
            marker="o",
            linewidth=2,
            label=TARGET_LABELS.get(target, target),
        )
    ax.set_xlabel("Character corruption rate (%)")
    ax.set_ylabel("Macro-F1")
    ax.set_ylim(0.45, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    ax.set_title("Text extraction robustness")
    save_figure(fig, "fig1_text_degradation_macro_f1")


def figure_maintenance_regression() -> None:
    df = pd.read_csv(OUT / "phase4_maintenance_risk_regression.csv")
    df = df[(df["model"] == "hist_gradient_boosting") & (df["eval_split"] == "test")].copy()
    df = df[df["feature_set"].isin(MAINTENANCE_FEATURE_SETS)]
    df["Feature set"] = df["feature_set"].map(FEATURE_LABELS)
    order = list(FEATURE_LABELS.values())
    df["Feature set"] = pd.Categorical(df["Feature set"], categories=order, ordered=True)
    df = df.sort_values("Feature set")
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.barh(df["Feature set"], df["r2"], color="#4d7c8a")
    ax.set_xlabel("Test R2")
    ax.set_xlim(0.90, 0.95)
    ax.grid(axis="x", alpha=0.25)
    ax.set_title("Maintenance-risk regression ablation")
    save_figure(fig, "fig2_maintenance_regression_r2")


def figure_client_holdout() -> None:
    df = pd.read_csv(OUT / "phase4_client_holdout_maintenance.csv")
    df = df[df["feature_set"].isin(MAINTENANCE_FEATURE_SETS)].copy()
    df["Feature set"] = df["feature_set"].map(FEATURE_LABELS)
    value_col = "decision_f1" if "decision_f1" in df.columns else "f1"
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    clients = sorted(df["holdout_client"].unique())
    feature_sets = [
        "Usage only",
        "Usage + true reports",
        "Usage + extracted reports",
    ]
    width = 0.24
    x = range(len(clients))
    for idx, feature_set in enumerate(feature_sets):
        values = []
        for client in clients:
            row = df[(df["holdout_client"] == client) & (df["Feature set"] == feature_set)]
            values.append(float(row[value_col].iloc[0]))
        positions = [p + (idx - 1) * width for p in x]
        ax.bar(positions, values, width=width, label=feature_set)
    ax.set_xticks(list(x))
    ax.set_xticklabels(clients)
    ax.set_xlabel("Held-out client")
    ax.set_ylabel("Decision F1")
    ax.set_ylim(0.75, 0.95)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Client-held-out decision performance")
    save_figure(fig, "fig3_client_holdout_decision_f1")

def main() -> None:
    ensure_dirs()
    table_dataset_summary()
    table_text_extraction()
    table_text_degradation()
    table_maintenance_regression()
    table_maintenance_decision()
    table_client_holdout()
    table_maintenance_uncertainty()
    figure_text_degradation()
    figure_maintenance_regression()
    figure_client_holdout()
    print(f"Publication artifacts written to: {ARTIFACTS}")


if __name__ == "__main__":
    main()
