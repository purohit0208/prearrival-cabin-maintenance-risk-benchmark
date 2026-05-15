# Maintenance-Only Results Summary

This is the current result summary for the MLWA manuscript after removing inventory forecasting from the main paper.

## Scope

Main paper includes:

- crew-report text extraction;
- report-only and usage-context maintenance-risk ablations;
- maintenance-risk index regression;
- validation-selected thresholded maintenance decision evaluation;
- text degradation robustness;
- client-held-out maintenance generalization.

Main paper excludes:

- same-flight inventory prediction, because it was leaky;
- corrected next-flight inventory prediction, because it was weak and distracts from the maintenance-risk contribution.

## Text Extraction

| Target | Test accuracy | Test macro-F1 | Gold accuracy | Gold macro-F1 |
|---|---:|---:|---:|---:|
| Component | 1.000 | 1.000 | 1.000 | 1.000 |
| Severity | 0.685 | 0.617 | 0.549 | 0.539 |
| Location | 0.996 | 0.996 | 0.998 | 0.997 |

Interpretation: component and location are easy in the synthetic template distribution; severity is the main extraction bottleneck.

## Text Degradation

| Target | Macro-F1 at 0% | Macro-F1 at 10% | Macro-F1 at 20% |
|---|---:|---:|---:|
| Component | 1.000 | 0.997 | 0.974 |
| Severity | 0.617 | 0.577 | 0.538 |
| Location | 0.996 | 0.932 | 0.848 |

Interpretation: the extraction layer is sensitive to report noise, especially for severity and location.

## Maintenance-Risk Regression

Histogram gradient boosting, test split:

| Feature set | MAE | RMSE | R2 |
|---|---:|---:|---:|
| True reports only | 0.206 | 0.254 | -1.887 |
| Extracted reports only | 0.202 | 0.251 | -1.804 |
| Usage only | 0.031 | 0.039 | 0.934 |
| Usage + true reports | 0.028 | 0.035 | 0.944 |
| Usage + extracted reports | 0.029 | 0.037 | 0.938 |

Interpretation: report features alone do not predict the temporally split synthetic risk index. They are useful as contextual augmentation to usage/cycle features, not as standalone maintenance-risk diagnostics. Extracted report features preserve most of the true-report regression gain when combined with usage/cycle features.

## Thresholded Maintenance Decision

Histogram gradient boosting with thresholds selected on validation data:

| Feature set | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| True reports only | 0.217 | 0.207 | 0.991 | 0.342 |
| Extracted reports only | 0.257 | 0.211 | 0.959 | 0.346 |
| Usage only | 0.945 | 0.829 | 0.923 | 0.873 |
| Usage + true reports | 0.928 | 0.749 | 0.977 | 0.848 |
| Usage + extracted reports | 0.944 | 0.860 | 0.868 | 0.864 |

Interpretation: report-only decisions are not useful. Thresholded decisions remain strong once usage/cycle context is included. The report feature sets should be interpreted alongside the regression results because F1 depends on threshold selection and target prevalence.

## Client-Held-Out Maintenance Generalization

| Feature set | Mean test R2 | Minimum test R2 | Mean decision F1 | Minimum decision F1 |
|---|---:|---:|---:|---:|
| True reports only | -2.164 | -3.586 | 0.340 | 0.234 |
| Extracted reports only | -2.244 | -3.407 | 0.333 | 0.238 |
| Usage only | 0.911 | 0.897 | 0.848 | 0.795 |
| Usage + true reports | 0.924 | 0.907 | 0.861 | 0.834 |
| Usage + extracted reports | 0.918 | 0.903 | 0.874 | 0.810 |

Interpretation: report-only models fail under client holdout. Extracted reports improve mean held-out decision F1 over usage-only when combined with usage/cycle context and retain most regression performance across client partitions. This is client-held-out generalization, not federated learning.

## Uncertainty And Effect Sizes

Bootstrap 95% confidence intervals for histogram gradient boosting test results show:

- usage-only R2: 0.934 (95% CI 0.925 to 0.941);
- usage plus true reports R2: 0.944 (95% CI 0.936 to 0.951), delta vs usage-only 0.011 (95% CI 0.007 to 0.014);
- usage plus extracted reports R2: 0.938 (95% CI 0.930 to 0.945), delta vs usage-only 0.004 (95% CI 0.001 to 0.007);
- usage plus extracted reports F1 delta vs usage-only: -0.009 (95% CI -0.035 to 0.016).

Interpretation: report features provide a small but measurable regression gain, while thresholded F1 differences are not clearly positive. The paper should emphasize risk-index information gain and decision comparability, not claim a large F1 improvement.
