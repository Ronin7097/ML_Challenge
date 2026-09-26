# Server approach review — 26 September 2026

This inventory covers accessible challenge work under `/home/jagat` and a shallow check of `/opt`, `/mnt`, and `/srv`. It is a review of separate experiments, **not** a claim that their code or outputs are part of the frozen DARPA submission. Scores from different cohorts, and local development scores versus public leaderboard scores, are not directly comparable. Test labels are withheld.

## Relevant folder names

| Server folder | Approach or role | Evidence available |
|---|---|---|
| `/home/jagat/amazon-ml-challenge-singleton-veto` | Granite multilingual dense retrieval, top-10 target-to-Source-1 owner scoring with LightGBM; then a learned singleton veto | Owner model: 98.054% on its 19,991-entity development cohort and **user-reported** 97.4% public score. Veto v1: 98.053% → 98.136% on its separate 4,014-entity audit; v2 and v3 failed their promotion gates. The veto test TSV is validated but has no reported leaderboard score. |
| `/home/jagat/codex-blocking-7UU4LCTs` | Structural address blocking and parsing comparisons: raw address, libpostal, DeepParse, TinyBERT, IndicBERT, ModernBERT | On a 10,000-link India sample, union of v7 retrieval with structural/name keys at cap 80 raised blocking recall from 92.44% to 96.06%, adding about 583,000 candidate pairs. On a 10,000-link US sample, 97.01% → 98.19%, adding about 239,000 pairs. These are retrieval results, not F₀.₅. In a separate sampled parser comparison, raw-address plus name-token blocking matched the best DeepParse/libpostal recall at the stated pair budget. |
| `/home/jagat/codex-blocking-7UU4LCTs/singleton_v5` | A1–A13 owner-decision experiments, including leave-owner-out, pair evidence, cost-aware veto, Source-1 context, link pruning, A8 unified winner decision, multilingual/generalization, runner-up rescue, augmentation, and reranking | A8 combines target runner-up and Source-1 incoming-link context with name/address pair evidence. Grouped out-of-fold development F₀.₅ is 98.591% versus 98.054% for the owner rule on the **same** 19,991 entities; paired gain CI 0.444–0.637 percentage points. Its full 1,732,544-row test TSV and candidate TSV passed the supplied validator; matching-ID existence also passed. No public A8 score found. A7 link pruning improved the same owner development score by about 0.139 points but was superseded by A8. |
| `/home/jagat/claude-merged-20260926` | A9: A8 winner decision augmented with Qwen3-Reranker scores for a bounded uncertain target union | Grouped out-of-fold development F₀.₅ 99.131% versus A8's 98.591% on the same development cohort. This is a development result after many explored variants; it is **not** a public or sealed-reserve score. No completed full-test A9 submission TSV found. |
| `/home/jagat/claude-supervisor-20260926` | Coordinates A9 test reranker scoring, A14 runner-up switch experiments, and France/normalization work | At inspection, 7 of 154 Qwen test-score shards existed. A14's best measured gain over A9 was small and its reported 95% interval crossed zero. Only a partial smoke output was found. |
| `/home/jagat/owner-repro-assets` | Reproducibility data/model snapshots for the owner work | Supporting assets, not a separate scored approach. |
| `/home/jagat/qwen3_reranker_lora_epoch01` | Qwen reranker adapter used by A9 | Model asset, not a separate evaluated pipeline. |
| `/home/jagat/.DARPA_submission_work` | Our frozen CatBoost pair/context pipeline | Full inference is running. The untouched 6,000-entity audit is 97.3744% macro F₀.₅; test F₀.₅ is unknown. |

## What is most useful for DARPA

1. **Target-first dense retrieval and owner competition** have the strongest retrieval evidence in the reviewed work: the owner approach reports 99.443% top-10 candidate recall on its development cohort. Our audit candidate recall is 97.486% on a different cohort, so this suggests an experiment rather than a measured head-to-head gain. Training and inference require a multilingual encoder and substantial GPU work.
2. **A8's contextual winner decision** is the clearest completed decision-layer result. Its features measure the second-best owner, winner margin, other targets won by the same Source 1, and structured name/address agreement. Our model already uses candidate and competing-Source-1 context, so the transferable test is an ablation on a fresh, shared, name-disjoint holdout, especially for target ownership and incoming-link evidence.
3. **Structural numeric/address keys** may recover candidate misses. Test their *incremental* true-link recovery after our own top-160 cap and report extra candidates and full macro F₀.₅. The parser comparison does not justify importing an NER model by itself.
4. **A9 Qwen reranking** is promising on development but is still scoring test shards. Wait for complete coverage, reproducibility, full validation, and a separate generalization check before considering it for a final submission. The repeated use of the development cohort can make a selected score optimistic.
5. **Singleton veto and runner-up switching** produce much smaller or unstable gains. Our current expected-F₀.₅ policy already considers the empty set, so a new veto needs a paired test against our own decisions before use.

Keep the current server inference and frozen output as a reproducible fallback. A8 is the strongest *validated full-test alternative file* found in this scan, but its public score is unknown and its own source/model/documentation would have to accompany any submission that uses it. Do not substitute a different TSV into the DARPA archive while documenting the CatBoost pipeline as its producer.

## Direct transfer checks on DARPA's frozen tuning outputs

We tested two small decision changes on our own 6,000-entity tuning split using the frozen context logits and all 960,000 candidate rows, without retraining or opening further audit labels:

- The selected 19,688 links had **zero** duplicate target assignments. Every selected link was also the highest-scoring edge for its target among the 6,000 sampled Source 1 entities. A one-owner-per-target or target-best-only filter therefore changed no tuning predictions and left macro F₀.₅ at 97.4422%.
- A score-only veto of one-link predictions, with raw-logit cutoffs from 0.5 to 5.0 selected within five folds, chose the no-op rule in every fold. It left tuning macro F₀.₅ at 97.4422%. There were only nine mistaken singleton entities, six with one predicted link, so a simple singleton veto has limited room to help.

In the first 150,000 rows of the *unlabelled* test prediction stream, 87 targets had multiple predicted Source 1 owners (88 extra assignments among 494,075 links). This shows that conflicts can occur at full scale, but it does not establish which owner is correct or the effect of a filter on test F₀.₅.

The next real improvement experiment should train target-competition and incoming-link features on a new name-disjoint development cohort, compare against the frozen CatBoost decision on identical entities, and separately measure extra true-link recall from structural retrieval after the 160-candidate cap. Only a validated gain on a fresh holdout would justify another full inference run.
