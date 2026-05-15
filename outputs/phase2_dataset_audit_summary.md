# Phase 2 Dataset Audit Summary

Flights: 5,267
Crew reports: 94,806
Inventory events: 126,408
Gold extraction labels: 1,200

## High-Priority Findings

- `maintenance_needed` is perfectly separated by `hazard_score`: 0.863 < 0.864. Binary maintenance classification would mostly reconstruct the generator rule.
- The old inventory task is not valid as independent forecasting evidence because it predicts `shortage_count` while `shortage_count` is included in the feature schema.
- Aggregated crew report count is constant per flight: [18]. This field should be dropped or explained.
- Aggregated inventory event count is constant per flight: [24]. This field should be dropped or explained.
- Gold-label `target_issue` is constant (`malfunction`), so it should not be used as a multiclass benchmark.
- The split order is monotonic train -> validation -> test for every tail, which supports a temporal-per-tail split claim.

## Recommended Technical Rebuild

1. Use `hazard_score` only as a synthetic maintenance risk index, not as a real hazard function.
2. Add text-extraction benchmarks for component, severity, and location from `report_text`.
3. Redesign inventory as next-flight or next-rotation shortage prediction using lagged features only.
4. Add local/client generalization tests instead of calling prediction averaging production federated learning.
5. Report all results as synthetic benchmark evidence with explicit limitations.
