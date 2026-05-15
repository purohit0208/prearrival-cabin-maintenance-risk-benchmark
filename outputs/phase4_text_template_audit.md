# Text Extraction Template Audit

Purpose: quantify whether the near-perfect component/location extraction results are a valid benchmark result or an artefact of deterministic synthetic templates.

## Exact Text Overlap

- Train rows: 63,918
- Test rows: 19,296
- Train unique report texts: 31,611
- Test unique report texts: 10,940
- Test unique report texts also seen in train: 2,618 (0.239)
- Test rows whose exact report text was seen in train: 10,810 (0.560)
- Test rows with report text not seen in train: 8,486

## Label Conflicts Per Unique Report Text

| target         |   unique_report_texts |   unique_report_texts_with_multiple_labels |   conflict_fraction |
|:---------------|----------------------:|-------------------------------------------:|--------------------:|
| component_code |                 44763 |                                          0 |            0        |
| severity       |                 44763 |                                       3407 |            0.076112 |
| location_hint  |                 44763 |                                          0 |            0        |

## Character TF-IDF Model On Seen vs Unseen Test Texts

| target         | eval_subset                 |     n |   accuracy |   macro_f1 |   weighted_f1 |
|:---------------|:----------------------------|------:|-----------:|-----------:|--------------:|
| component_code | test_all_rows               | 19296 |   1        |   1        |      1        |
| component_code | test_rows_seen_in_train     | 10810 |   1        |   1        |      1        |
| component_code | test_rows_not_seen_in_train |  8486 |   1        |   1        |      1        |
| severity       | test_all_rows               | 19296 |   0.684909 |   0.617416 |      0.689359 |
| severity       | test_rows_seen_in_train     | 10810 |   0.692969 |   0.626406 |      0.695802 |
| severity       | test_rows_not_seen_in_train |  8486 |   0.674641 |   0.604171 |      0.67968  |
| location_hint  | test_all_rows               | 19296 |   0.995647 |   0.995933 |      0.995655 |
| location_hint  | test_rows_seen_in_train     | 10810 |   0.999167 |   0.999302 |      0.999168 |
| location_hint  | test_rows_not_seen_in_train |  8486 |   0.991162 |   0.990709 |      0.991213 |

## Interpretation

- Component and location labels are deterministic or nearly deterministic from surface text in this synthetic corpus.
- Component has no label conflicts per unique report text and remains perfect even on test rows whose exact report text was not seen during training.
- Location has no label conflicts per unique report text, but no-overlap test performance drops below the all-test result.
- Severity has many conflicts per unique report text and is therefore the more meaningful extraction bottleneck.
- The manuscript should present component and location extraction as synthetic-template sanity checks, not as evidence of difficult aviation NLP solved by the model.