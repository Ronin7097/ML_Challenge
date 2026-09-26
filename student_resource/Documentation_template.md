# ML Challenge 2026: Business Entity Resolution

**Team Name:** DARPA
**Submission Date:** 26 September 2026

## 1. Executive Summary

The improved matcher combines a boosted pair classifier, candidate context, and comparisons against competing Source 1 businesses. Its frozen macro F₀.₅ is **0.973744** on 6,000 untouched audit entities. The requested 0.98 target was not reached. These are local validation results, not leaderboard results. Only the supplied challenge data is used.

## 2. Methodology and Data

Training contains 2,206,821 Source 1 records. Target sources contain noisy names and addresses, missing addresses, transliterations, Indic scripts, reordered words, and near misses. Test additionally includes France, which is absent from training. Country is an open label used for comparisons; no country is excluded from prediction.

Entity-disjoint development splits contain 80,000 pair-fitting entities, 30,000 context-fitting entities, 6,000 tuning entities, and 6,000 audit entities. The original tuning and audit IDs were preserved when expanding training data. Audit labels were excluded from training, early stopping, error inspection, and decision selection. The final model and decision rule were frozen before audit evaluation.

## 3. Candidate Generation

Country-aware retrieval combines name and address tokens, token pairs, normalized whole fields, consonants, and numerical address keys. Bounded retrieval and similarity reranking retain up to 160 candidates per entity. Every final match is drawn from the recorded candidate set.

Audit candidate recall is **97.486%**. With perfect decisions on these candidates, the audit macro F₀.₅ ceiling is **99.158%**. Unretrieved true links remain in all evaluation denominators.

## 4. Matching Model

The pair classifier uses 107 numerical features describing character/token overlap, fuzzy agreement, legal suffixes, missing addresses, alphanumeric components, and approximate transliteration/sound similarities. CatBoost training starts on a smaller fitting sample, then continues on the larger fitting set, producing 2,999 trees.

The context classifier uses 156 features: the 107 pair features, 23 summaries of other candidates' probabilities and sources, and 26 features comparing competing Source 1 businesses, name rarity, and numerical errors. It has 1,800 depth-7 trees and is fitted on the separate 30,000-entity context split.

The reference index contains unlabelled Source 1 input fields from the corresponding train or test data. Validation fields can participate in this index, matching deployment behavior, but their labels never enter features. Transliteration rules are generated from Python's bundled Unicode character names. No external business lookup, geocoding, or pretrained weights are used.

For each entity, the decision rule compares the empty set with successive prefixes of candidates ranked by context probability. A prefix of size k has utility `1.25 × sum(p[:k]) / (k + 0.25 × sum(p))`; the empty-set utility is `product(1-p)`. This approximates expected F₀.₅. Actual evaluation uses the official macro metric, including singletons and unretrieved truth.

## 5. Results and Limitations

| Configuration | Tuning macro F₀.₅ | Audit macro F₀.₅ |
|---|---:|---:|
| Original logistic baseline | 82.87% | 83.39% |
| Starting boosted model | 94.15% | 94.20% |
| Improved frozen model | **97.44%** | **97.37%** |

The improved model's audit link precision is **99.43%**, link recall **93.63%**, and singleton accuracy **97.43%**. Compared with the starting boosted model, audit macro F₀.₅ improves by **3.18 percentage points** on identical entities; the approximate paired 95% interval is **2.85–3.50 points**.

Audit F₀.₅ is 96.70% for India and 97.82% for the US. There is no labelled France audit, so these results do not establish French test accuracy. Candidate misses and difficult aliases/numerical near misses still limit performance. The audit was not used to make further model changes.

## 6. Reproduction and Submission Status

`code/business_entity_resolution/README.md` gives the training, frozen audit, and portable-inference commands. `models/` contains the portable pair/context models, checksums, settings, and validation results. Python/C++ parity was verified on 3,200 tuning pairs with zero raw-score error and identical final decisions.

The packaged model passed a 128-row test smoke run and the official submission validator, including target-ID existence checks. Eleven Python unit tests and the C++ feature checks passed. Smoke outputs cover India, the US, and France; they verify execution and formatting, not test accuracy.

To regenerate the submitted output from the extracted archive, place the supplied `dataset/` directory next to `code/` and run from the archive root:

```sh
python3 code/business_entity_resolution/src/predict.py . output_reproduced --threads 8
```

Inference requires a C++17 compiler and Python 3; third-party Python packages are needed only for training and development evaluation. The runner preserves existing outputs, validates completed files, and records model checksums. `--max-queries 128` produces only a smoke-test prefix.

The implementation and packaged models use the repository's MIT license.
