# Pre-Arrival Cabin Report Maintenance-Risk Benchmark

This repository contains the synthetic dataset, reproducible scripts, and generated result tables/figures for the manuscript:

**A Leakage-Aware Synthetic Benchmark for Pre-Arrival Cabin Report Intelligence and Maintenance-Risk Decision Support**

The benchmark evaluates how information extracted from pre-arrival cabin-report text propagates into downstream synthetic maintenance-risk decision support.

## Contents

- `data/synthetic_intact_llm_v2_realistic/`: synthetic benchmark CSV/JSON files.
- `src/`: reproducible audit, extraction, robustness, maintenance-risk, uncertainty, and artifact-building scripts.
- `outputs/`: generated audit outputs, result CSV files, and publication tables/figures.
- `runs/RUN_20260302_093458/feature_schema.json`: legacy feature-schema file used only by the leakage audit.

## Main Dataset Files

- `flights.csv`: per-flight synthetic aircraft, route, weather, usage, and component-age records.
- `crew_reports.csv`: synthetic pre-arrival cabin reports with structured labels.
- `maintenance_labels.csv`: synthetic `hazard_score` and threshold-derived `maintenance_needed` labels.
- `gold_standard_labels.csv`: held-out extraction-label subset.
- `split.csv`: train/validation/test split.
- `schema.json`, `data_card.json`, `dataset_stats.json`, `clients.json`: dataset metadata.

The data are fully synthetic and contain no real airline operational data.

## Reproducing Results

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run the main pipeline:

```powershell
python .\src\audit_dataset.py
python .\src\run_text_extraction_baselines.py
python .\src\run_transformer_severity_baseline.py
python .\src\audit_text_extraction_template_effects.py
python .\src\run_text_degradation.py
python .\src\run_maintenance_risk_benchmarks.py
python .\src\run_maintenance_uncertainty.py
python .\src\run_client_holdout_benchmarks.py
python .\src\build_publication_artifacts.py
```

The transformer baseline downloads `sentence-transformers/all-MiniLM-L6-v2` from Hugging Face through the `transformers` library.

## Scientific Boundary

This repository supports reproducible synthetic benchmark evidence only. It does not validate a deployable aviation AI system, certified trustworthy AI, operational generalization, or real airline maintenance performance.

## License

Code is released under the MIT License. The synthetic dataset and generated tables/figures are released under CC BY 4.0; see `DATA_LICENSE.md`.
