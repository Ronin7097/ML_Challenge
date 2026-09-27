# ML Challenge 2026: Business Entity Resolution

**Team Name:** DARPA
**Submission Date:** 27 September 2026

## 1. Executive Summary

We match Source 2 and Source 3 records to canonical Source 1 businesses using a
country-partitioned dense retrieval model, a lexical/numeric pair classifier,
and an incoming-link context classifier. The final policy is a frozen 0.62
context-probability threshold. Each target is assigned to at most one Source 1
business; a Source 1 business may receive multiple targets.

On a fresh, name-group-disjoint 7,882-entity tuning cohort, this policy scores
**0.982959 macro F₀.₅** versus **0.977520** for our earlier frozen CatBoost
submission on the identical cohort. The threshold was selected on this tuning
cohort. The already frozen policy then scored **0.984081** on the separate
7,909-entity sealed reserve, versus **0.980752** for our earlier CatBoost
submission on the same reserve. These are local training-data holdouts, not
Portal or test scores. Test labels are not available locally.

## 2. Data and Evaluation Design

Only the challenge's supplied training records and labels fit the encoder and
classifiers. The Source 1 training table has 2,206,821 businesses. We partition
whole country/normalized-name groups into disjoint encoder, pair-fitting,
context-fitting, tuning, and sealed-reserve roles. Historical DARPA development
name groups are excluded from the new tuning and reserve roles. There are
23,770 pair-fitting, 23,746 context-fitting, 7,882 tuning, and 7,909 reserve
Source 1 entities. Every role's targets remain in the full retrieval pool; a
role's labels are used only in its authorized training or evaluation stage.

The test set includes France, which is absent from labeled training. Country
values remain open strings; no fixed list of countries is assumed in the
pipeline. No external business identity lookup, geocoding, or cross-team
predictions or trained weights are used.

## 3. Candidate Generation

The encoder starts from IBM Granite embedding 97M multilingual r2, an
Apache-2.0 model pinned to revision
`835ad14087e140460703cf0fae09f97d469d65c2`. It is fine-tuned on one
million positive target/owner pairs from the encoder-fitting role. The saved
384-dimensional normalized embeddings are searched within country using FAISS.
Each target retains its ten highest-scoring distinct Source 1 owners. These
complete ten-owner groups are the **actual pair-classifier candidate set**.
The submitted `candidate_pairs.tsv` reverses this incidence list into one row
per Source 1 entity; it is not shortened after matching decisions.

On the 27,261 positive tuning links, top-ten true-owner retrieval recall is
**0.995378** (27,135 recovered, 126 missed). This is retrieval recall, not
F₀.₅. All 10,320,219 training targets were retrieved for evaluation, including
unrelated records that can become false positives. The encoder treats the
country as a blocking key; French test records are retrieved within France
using the same learned representation.

## 4. Matching Model and Decisions

The pair LightGBM evaluates all ten owners per target with 76 features:
normalized name/address similarities, exact and fuzzy token comparison,
character n-grams, legal suffix and abbreviation normalization, address
numbers, missing fields, dense similarity/rank, and each owner's standing
against the other nine owners. All scripts are retained during normalization;
Latin diacritics are folded without deleting Indic characters.

The highest pair-probability owner becomes the provisional winner for each
target. Five LightGBM context models then average their probabilities using
90 features: the pair features, score distribution over the ten rivals, and
other target links provisionally assigned to the same owner. The incoming-link
features exclude the link being scored. An average context probability of at
least **0.62** accepts the target; otherwise the target remains unmatched.
Only targets with an owner in the relevant fitting/evaluation role are used
for that role's complete incoming-context evaluation. Final inference covers
the full test owner and target pools.

## 5. Ablations and Model Selection

The 300-step encoder pilot reached 0.989546 top-ten retrieval recall. A
label-confirmed hard-negative continuation reached 0.991563, including
same-name different-owner negatives. The full one-million-pair encoder reached
0.995378, so it was selected at the same ten-candidate budget. A learned
ten-owner selector with an explicit no-match option scored 0.979841 tuning
macro F₀.₅ and was rejected. A fine-tuned Qwen3-Reranker-0.6B selectively
rescored all 6,110 India and 7,447 US routed tuning groups; its best tested
override policy scored 0.974163, adding false positives, so it was rejected.
These variants and the research leads are documented in `EXPERIMENTS.md`.

The selected model has 26,506 true-positive links, 206 false positives, and
755 missed links on the new tuning cohort. The paired improvement over the
frozen CatBoost baseline is **+0.005439** macro F₀.₅, with a 2,000-replicate
name-group bootstrap interval of **[+0.003417, +0.007640]**. The tuning
singleton accuracy is 0.977887. The frozen policy and model SHA-256 hashes
are in `reports/frozen_policy.json`.

After policy selection, the reserve was evaluated once without changing the
model or threshold. It has 26,773 true-positive links, 194 false positives,
and 732 missed links. The paired reserve improvement is **+0.003329** macro
F₀.₅, with a 2,000-replicate name-group bootstrap interval of
**[+0.001341, +0.005227]**. The reserve result is in
`reports/reserve_paired_f05.json`. No reserve result was used to retune the
decision threshold.

## 6. Reproduction and Submission Validation

The archive's `code/business_entity_resolution/README.md` gives the exact
environment and commands for preparing the supplied data, retrieving complete
test candidates, calculating every pair and context feature, and producing
both TSVs. The archive includes the fine-tuned Granite weights/tokenizer, the
pair and five context LightGBM models, pinned dependencies, and source code.
The Granite base and derivative weights use Apache-2.0; our code uses MIT.
The selected model is well below the 8-billion-parameter limit.

The frozen full-test run covered **1,732,544 Source 1 entities** and
**9,969,589 targets** across France, India, and the US. It produced
**6,008,048 predicted links** and **99,695,890 actual scored candidate pairs**,
ten per target. The output manifest records SHA-256 identities for both TSVs.
The streaming validator passed on all matching and candidate rows, and the
official validator passed with matching-ID existence checked against the full
test target set. The separate validation report records both results. These checks establish submission
integrity, not withheld test F₀.₅.
