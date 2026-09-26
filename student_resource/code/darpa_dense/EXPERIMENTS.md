# DARPA dense-owner experiment log

All numbers below are from supplied competition training data unless labelled
otherwise. The new tune cohort has 7,882 Source 1 entities and excludes name
groups used in the earlier DARPA development sample. Its 7,909-entity reserve
cohort remains sealed. Local tune scores, the earlier audit, and Portal scores
measure different populations.

| Approach | Same-cohort result | Outcome |
| --- | ---: | --- |
| Frozen CatBoost baseline | 0.9775204 macro F0.5 on new tune; historical audit 0.9737442; user-reported Portal 0.9608 | Validated fallback archive retained unchanged |
| 300-step Granite embedding pilot, one positive target per owner | True-owner retrieval recall 0.9895455 at top 10 of 27,261 tune links | Below predefined 0.99 retrieval gate; no full-target pilot inference |
| HN1: 100,000 positive and 100,000 label-confirmed wrong-owner pairs; 12,217 negatives have the same normalized name | True-owner retrieval recall 0.9915630 at top 10; 55 more links than pilot | Full train target retrieval completed; not an F0.5 score |
| One-million-pair full Granite encoder | True-owner retrieval recall 0.9953780 at top 10; 104 more links than HN1 | Chosen for first full owner model; retrieval is not F0.5 |
| Full encoder + pair LightGBM + incoming-context LightGBM | 0.9829591 macro F0.5 on same tune; +0.0054387 versus frozen baseline, paired name-group bootstrap interval [+0.0034172,+0.0076403] | Current best measured challenger. Threshold 0.62 was tuned on this cohort, so independent performance remains unmeasured |
| SEL1: second rival-aware LightGBM chooses among ten owners and can reject all | 0.9798411 macro F0.5 on same tune; +0.0023207 versus frozen baseline but 0.0031180 below current challenger | Not promoted. Wrong-owner rescue versus regression analysis may inform a narrower combined policy |
| Qwen3-Reranker-0.6B LoRA on 200,000 mined pairs | 3,125 optimizer steps complete; no F0.5 yet | A 2-step GPU smoke previously passed. Bounded group scoring on Slurm reached 512 India groups then hit the account's NFS disk quota; inputs copied and hash-verified to A6000 for rerun |

The full encoder scored all 10,320,219 training targets against every Source 1
record in their country. Feature extraction retained all ten rival owners for
each of 1,957,017 selected target groups, producing 19,570,170 candidate-pair
rows. The complete target pool matters because unrelated target records can be
false positives for tune owners. [The paired report](reports/full_owner_paired_tune.json)
contains country and name-frequency strata; [the selector report](reports/sel1_paired_tune.json)
shows its separate result. Retrieval reports are in `reports/`.

The public dense retrieval, hard-negative, owner-selection, and reranking ideas
were used as research directions. No other team's private predictions or
trained weights were copied into this experiment. See
`PUBLIC_APPROACH_REVIEW.md` and the repository's
`research/ONLINE_APPROACH_UPGRADES_20260927.md` for sources and limitations.

Next gates: verify Qwen on bounded groups, test whether it improves the same
tune cohort, freeze the selected policy, then evaluate once on the sealed reserve.
Only then should a new full test submission replace the validated fallback.
