# Dataset Card

## Name

`synthetic_intact_llm_v2_realistic`

## Description

Synthetic aviation-turnaround benchmark data for studying pre-arrival cabin-report extraction and downstream maintenance-risk decision support.

## Intended Use

The dataset is intended for:

- text extraction from synthetic cabin reports;
- leakage-aware benchmark design;
- maintenance-risk index regression;
- validation-selected thresholded maintenance-decision evaluation;
- robustness and client-partition generalization checks.

## Not Intended For

The dataset must not be used as evidence of:

- real airline maintenance performance;
- operational safety assurance;
- deployable or certified aviation AI;
- autonomous maintenance decision-making.

## Dataset Composition

- Flights: 5,267
- Crew-report rows: 94,806
- Inventory-event rows: 126,408
- Gold extraction-label rows: 1,200
- Synthetic clients: 4
- Synthetic tails: 300

## Key Validity Notes

- `maintenance_needed` is threshold-derived from `hazard_score`; direct binary training on `maintenance_needed` is not treated as an independent learning task.
- Component and location extraction are deterministic or near-deterministic in the synthetic text-template distribution.
- Severity extraction is the main non-deterministic text-extraction bottleneck.
- The train/validation/test split is temporal by tail.

## Privacy

The dataset is fully synthetic and contains no real airline operational data.

## License

Dataset files are released under CC BY 4.0. See `DATA_LICENSE.md`.
