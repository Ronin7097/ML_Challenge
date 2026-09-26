# Experiment history and improvement opportunities

This is the development record for DARPA's ML Challenge 2026 entity resolver. Scores below are local **macro F₀.₅ over Source 1 entities**, including true singletons and true links missed by candidate retrieval. They are not leaderboard results. Except where marked **audit**, every score is from the same 6,000-entity tuning set; these numbers guided model selection and must not be treated as independent generalization estimates. Different rows sometimes change multiple components, so differences are not controlled ablations.

## What we tried

| Stage | Main change | Tuning F₀.₅ | Outcome |
|---|---|---:|---|
| Original logistic baseline | Rare-token name/address blocking, top 80 candidates, 20 hand-built pair features, precision-focused threshold and rules | 82.87% | Starting point; sampled candidate recall was 91.5%. |
| Boosted pair model v1 | CatBoost on 76 pair features, top 160 candidates | 94.15% | Large improvement; tuning candidate recall was 96.01%. |
| Boosted pair model v2 | Improved candidate retrieval with the same 76-feature family, deeper pair model | 94.82% | Tuning candidate recall rose to 97.47%. |
| Context model v2 | Added 23 within-query features to the 76 pair features: other candidates' scores, ranks, and source agreement | 95.46% | Showed that decisions benefit from seeing the candidate set. |
| Boosted pair model v3 | Added 31 features for transliteration, names, and alphanumeric/numerical addresses (107 pair features total) | 95.83% | Improved pair scoring. |
| Context model v3 | 107 pair plus 23 candidate-context features; no competing Source 1 comparisons | 96.23% | Better than the pair threshold alone. |
| Competing-reference context, depth 6 | Added 14 features comparing a candidate with other unlabelled Source 1 records; 144 context features total | 96.81% | Reduced confusion among similar businesses. |
| Competing-reference context, depth 7 | Same 144 features, depth 7 and entity weight 0.5 | 96.91% | Best of the smaller-reference variants. |
| Extra phonetic blocking v4 | Generated more candidates from approximate sound keys | 95.80% at pair stage | Rejected: candidate recall fell from 97.47% to 97.37%, and pair F₀.₅ was below v3's 95.83%. |
| Larger pair fit | Continued the v3 pair model on 80,000 fitting entities; 2,999 trees | 96.05% | Supplied scores to the final context model. |
| Larger context fit, global threshold | 30,000 separate context-fit entities; 107 pair + 23 context + 26 competing-reference/numeric features; 1,800 depth-7 trees | 97.31% | Strongest global-threshold variant. |
| Larger context fit, per-entity set choice | Same model; chose an empty set or a top-k prefix by approximate expected F₀.₅ | **97.44%** | Frozen final decision rule. Untouched audit: **97.37%**. |

The earlier 144-feature reference models had 14 competing-reference features. The final 156-feature model added 12 numerical comparison features while also using more fitting data. These effects were not isolated. A score-blend experiment using older model outputs did not beat the selected context model, so no blend is deployed. On the smaller context v3, a probability scale/shift grid for the per-entity rule raised tuning F₀.₅ only from 96.227% to about 96.276%; the rule became useful with the larger model.

For the final model, a grid of probability scales and shifts gave 97.442951% at scale 1.25, shift 0, versus 97.442214% with the raw model logits (scale 1, shift 0). We selected the simpler raw-logit version because the tuning difference is 0.000737 percentage points and it has no extra fitted calibration parameter. This choice was frozen before audit evaluation.

## Final evaluation and safeguards

The 122,000 sampled Source 1 entities are disjoint: 80,000 for pair fitting, 30,000 for context fitting, 6,000 for tuning, and 6,000 for audit. The original tuning and audit IDs were preserved when fitting data grew. Candidate generation and model selection used only the supplied challenge records; no business identity API, geocoder, external database, or pretrained model was used. Unlabelled held-out Source 1 fields can enter the competing-reference index because the same kind of index is available at test time; held-out labels never enter those features.

The final untouched audit macro F₀.₅ was **97.3744%** (precision 99.4272%, recall 93.6284%, singleton accuracy 97.4286%). The archived v1 boosted model scored **94.1992%** on those same audit entities, a paired gain of **3.1752 percentage points** (approximate 95% interval 2.8493–3.5012). The original logistic decisions scored 83.3946%. Audit candidate recall was 97.4860%, and the perfect-decision ceiling on the retrieved candidates was 99.1583% macro F₀.₅. No test-set F₀.₅ can be calculated locally because the test labels are withheld.

The model is packaged in `models/` with its settings, checksums, tuning/audit metrics, and paired comparison. `README.md` gives end-to-end reproduction commands. Python and portable C++ scores/decisions matched on 3,200 tuning pairs. The packaged matcher passed a real 128-query test smoke run and the official format and target-ID checks. The submission's full outputs must receive the same checks before upload.

## Where to improve next

1. **Candidate misses:** About 2.51% of audit true links are absent from the top 160 candidates. Inspect missed links on a new development split, especially aliases and severe transliteration, then try recall improvements without adding common-name false candidates. The v4 sound-blocking result shows that more keys do not automatically help.
2. **Hard near matches:** Current false positives include similar names and addresses with conflicting street, unit, or business identity details. Explore carefully validated structural address evidence and confidence calibration on fresh, entity-disjoint data.
3. **Generalization:** Training and labelled audit contain US and India, while test also contains France. Build an additional labelled validation set if the challenge ever provides one; current French smoke checks establish only that the pipeline runs and emits valid IDs.
4. **Inference time:** Competing-reference retrieval is expensive on the full test set. A bounded cache preserves identical 128-query outputs, but larger improvements should be profiled and verified for exact prediction parity before use.

Do not tune further on the 6,000-entity audit and still call it untouched. Use a fresh holdout or cross-validation for any new experiments.
