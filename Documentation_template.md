# ML Challenge 2026: Business Entity Resolution

**Team:** DARPA

**Date:** 27 September 2026

## 1. Executive Summary

The submitted model matches Source 2 and Source 3 business records to the
deduplicated Source 1 reference. A fine-tuned multilingual encoder retrieves ten
candidate Source 1 owners per target. A LightGBM pair classifier selects an owner,
and base/rich context ensembles decide whether to accept that link. Each target
has at most one accepted owner; each Source 1 can have zero or multiple targets.

The team reports a **0.970 Portal score** for these predictions. The output files
cover all **1,732,544 test Source 1 entities**, including France, with **5,958,911
predicted links**. Portal feedback and local evaluation scores are distinct.

## 2. Methodology and Data

Only supplied competition records and training labels were used for fitting.
No external business database, geocoding service, identity lookup, web-derived
business augmentation or entity-resolution API is used. Generic AnyAscii 0.3.3
transliteration is a software dependency; original-script comparisons are also
retained. The pipeline runs offline with the packaged model files.

Training Source 1 records are split by normalized country/name groups. Encoder,
pair, context, development-tune and reserve roles are separated. Historical
development groups are excluded from new evaluation roles. The encoder was
fine-tuned on one million supplied positive training pairs. Same-name and
identical-text in-batch false negatives are masked during contrastive training.
Pair and context classifiers fit separate roles; tune/reserve target outcomes
do not fit the submitted rich head. Test labels are unavailable and were not
accessed. IDs locate and export records rather than act as learned features.

Country partitions are discovered from the data. There is no US/India-only
filter or closed country one-hot vocabulary. The unseen test country France is
included with the same inference policy.

## 3. Candidate Generation / Blocking

Source 1 and target names, addresses and country strings are encoded with the
fine-tuned IBM Granite embedding 97M multilingual r2 model. Normalized vectors
use CLS pooling and 384 dimensions; text is truncated to 96 tokens. Within each
country, batched GPU inner products retain the ten strongest owner candidates
for every target. These edges are inverted into Source 1 candidate lists.

The candidate export contains the **exact 99,695,890 edges scored by the pair
matching model**, including subsequently rejected rivals. The richer acceptance
stage preserves that file. It has one row per Source 1; every accepted link is a
member of its candidate list. It is not trimmed after model scoring.

| Candidate count per Source 1 | Value |
| --- | ---: |
| Mean | 57.5431 |
| Median | 49 |
| 95th percentile | 128 |
| 99th percentile | 187 |
| Maximum | 1,219 |
| Empty candidate lists | 1 |

Ten is the shortlist length per target, not per Source 1. The retained matching
pairs represent 0.00148256% of the same-country Cartesian pair space. This is a
matching-workload reduction: current retrieval still computes exhaustive
same-country embedding similarities. It is not claimed to scale unchanged to
billions of records. Candidate size affects final ranking; no numerical
accuracy/candidate-cost trade-off is specified, and optimality is not claimed.

## 4. Model Architecture and Feature Engineering

The 76-feature LightGBM pair classifier uses name/address similarities, legal
suffix normalization, token overlap, numeric disagreements, name frequency,
embedding evidence and comparisons among candidate owners. It scores all ten
candidates and selects the strongest owner.

Five 90-feature base-context classifiers use selected-pair evidence plus other
incoming target evidence. Five rich-context classifiers use **179 features**:
87 retained pair/context features and 92 additional features covering
transliteration, missing/extra name tokens, numeric address edits, a competing
owner and agreement with another incoming target. The supporting target always
excludes the current record. Complete incoming winners contribute to context.
The rich ensemble fits 110,480 context rows; an internal name-group split selects
730 boosting rounds per model.

The frozen policy is:

- Base probability strictly between 0.02 and 0.999: accept when the rich ensemble
  mean is at least 0.64.
- Otherwise: retain the base-context decision at threshold 0.62.

The producing pair model is rescored over complete routed rival groups to
recover the second-highest probability exactly. Recomputed owners must match
the cached owners. Subtraction of rounded cached scores is not used as a
substitute for this probability.

The encoder contains **97,441,152 stored parameters** and derives from the
Apache-2.0 Granite base revision `835ad14087e140460703cf0fae09f97d469d65c2`.
The encoder plus small tree ensembles is below 8 billion parameters. The
fine-tuned encoder remains Apache-2.0; DARPA source and original tree models
are MIT. Upstream license text, model card and modification notice are included.

## 5. Results and Limitations

| Evaluation | Entities | Macro F0.5 |
| --- | ---: | ---: |
| Development tune | 7,882 | 0.9846719728 |
| Previously used reserve | 7,909 | 0.9864138055 |
| Portal, team-reported | Public leaderboard subset | 0.970 |

Development link counts are TP 26,543, FP 179 and FN 718. Reserve counts are
TP 26,806, FP 176 and FN 699. The metric is averaged per Source 1, including
singletons: an empty truth with an empty prediction scores 1; an empty truth
with any prediction scores 0. Otherwise the per-entity count formula is
`1.25 * TP / (0.25 * true_link_count + predicted_link_count)`.

The tune cohort has supported multiple experiments. The reserve was also used
previously, so it is supporting evidence rather than a fresh blind audit;
reserve outcomes did not fit this rich head or select its threshold. Included
paired-bootstrap reports do not correct for adaptive experimentation. Local
scores do not establish a private-leaderboard score. There are no labeled
France validation examples, and country transfer remains uncertain.

## 6. Reproduction and Submission Validation

`code/business_entity_resolution/README.md` gives exact commands for data
preparation, embedding, retrieval, feature construction, both matching stages,
export and official validation. Pinned base and rich-stage requirements,
producing checkpoints, policy, split identifiers and source are included.
Regenerating predictions requires supplied data and the packaged weights, not
remote business or model services.

Both output TSVs passed the unmodified official validator with explicit
`--matching`, `--candidate`, `--test-dir` and `--check-ids`, without warnings.
The audit verifies valid target IDs, unique rows and lists, complete Source 1
coverage, and match membership in candidates. Coverage by country is France
259,452; India 809,986; US 663,106. The matching file contains 98,268 empty rows.
Checksums bind the delivered files to that audited run. Model and source hashes,
local evaluation reports and exact feature-cache parity evidence are included.

Upload `output/matching_results.tsv` to the live Portal. Submit
`DARPA_submission.zip` with both TSVs, code, models and this methodology document
for final review. The matching TSV in the ZIP is identical to the Portal file.
