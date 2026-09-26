# F₀.₅ improvement results

The improved model achieved **97.37% macro F₀.₅ on an untouched 6,000-entity audit**. The requested **98% target remains unmet**, with a gap of 0.63 percentage points. Its tuning score is 97.44%. No leaderboard score is available.

| Metric on the same audit entities | Starting boosted model | Improved frozen model |
|---|---:|---:|
| Macro F₀.₅ | 94.20% | **97.37%** |
| Link precision | 98.03% | **99.43%** |
| Link recall | 88.05% | **93.63%** |
| Correct empty singleton predictions | 92.57% | **97.43%** |
| True positive links | 18,282 | **19,441** |
| False positive links | 367 | **112** |
| False negative links, including retrieval misses | 2,482 | **1,323** |

The paired macro F₀.₅ gain is **3.18 percentage points** (approximate 95% interval: 2.85–3.50 points). The original logistic baseline scored 83.39% on these audit entities. The improved score has standard error 0.126 percentage points. Precision is reported separately and must not be mistaken for the requested F₀.₅.

## Changes

- Expanded fitting data to 80,000 pair-model entities and 30,000 separate context-model entities while preserving 6,000 tuning and 6,000 audit IDs.
- Added approximate cross-script name features, alphanumeric address comparisons, competing Source 1 evidence, name rarity, and digit omission/substitution features.
- Fitted a 107-feature pair model followed by a 156-feature context model.
- Selected a per-entity prediction-set rule on tuning data using approximate expected F₀.₅, with an explicit empty-set option.
- Packaged portable C++ inference, checksums, audit results, and reproduction commands.

The configuration was frozen before auditing; no subsequent decision tuning used audit results. The same audit entities were then scored with the archived starting boosted model solely for the paired comparison.

## Validation and limits

Python metric, context, and decision-policy checks passed. Portable C++ inference matched Python raw scores exactly and produced identical decisions on 3,200 tuning pairs from 20 entities. Split checks confirmed unique Source 1 IDs and no entity overlap across fitting, tuning, and audit.

The packaged model also passed an end-to-end 128-row test prefix using the full target and reference indexes: 392 matches from 20,480 candidates, covering 57 India, 49 US, and 22 France rows. The official submission validator passed with ID-existence checks against all 9,969,589 target IDs. Eleven Python unit tests and the C++ feature checks passed. This validates execution and formatting, not test-set accuracy; the prefix is not a complete submission. Details are saved in `models/smoke_test.json`.

Audit candidate recall is 97.486%; the perfect-classification ceiling for the retrieved candidates is 99.158% macro F₀.₅. India audit F₀.₅ is 96.70% and US audit F₀.₅ is 97.82%. France has no labelled validation examples, so test performance there is unverified. The metric includes every audit entity, correctly empty singletons, and truth links missed by retrieval.

The reference index uses unlabelled Source 1 fields, including held-out input fields, as it does at test time. It does not use held-out labels. All business data and training labels come from the challenge files.

## Artifacts and running predictions

- `code/business_entity_resolution/models/validation.json`: frozen tuning/audit metrics and parity results.
- `code/business_entity_resolution/models/audit_comparison.json`: paired comparison with the starting boosted model.
- `code/business_entity_resolution/models/config.json`: model checksums, feature counts, split sizes, and decision rule.
- `code/business_entity_resolution/README.md`: inference and training commands.

```sh
python3 student_resource/code/business_entity_resolution/src/predict.py \
  student_resource student_resource/output_advanced --threads 6
```

The existing `student_resource/output/` TSVs and `ML_Challenge_submission.zip` remain the earlier logistic baseline. Full improved test predictions and a new submission archive have not been generated. A smoke-test prefix is not a submission.
