# Reproducibility Notes

## Environment

The scripts were run with Python 3.11 on Windows. The core dependencies are listed in `requirements.txt`.

## Main Commands

Run from the repository root:

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

## Expected Outputs

The scripts write result CSV/JSON/Markdown files under `outputs/`, including:

- `phase2_dataset_audit.json`
- `phase4_text_extraction_results.csv`
- `phase4_transformer_severity_results.csv`
- `phase4_text_template_audit.md`
- `phase4_text_degradation.csv`
- `phase4_maintenance_risk_regression.csv`
- `phase4_maintenance_decision_thresholds.csv`
- `phase4_maintenance_uncertainty.csv`
- `phase4_client_holdout_maintenance_summary.csv`
- `outputs/publication_artifacts/tables/`
- `outputs/publication_artifacts/figures/`

## Transformer Baseline

`src/run_transformer_severity_baseline.py` uses mean-pooled embeddings from `sentence-transformers/all-MiniLM-L6-v2` through Hugging Face `transformers`, followed by a balanced logistic-regression classifier. The model weights are downloaded at runtime unless already cached.

## Path Configuration

By default, scripts look for:

`data/synthetic_intact_llm_v2_realistic`

Optional environment variables:

- `INTACT_DATA_DIR`: override dataset directory.
- `INTACT_RUN_DIR`: override legacy run directory used by the leakage audit.
- `INTACT_OUT_DIR`: override output directory.
