# DARPA dense-owner experiment log

All numbers below are from supplied competition training data unless labelled
otherwise. The new tune cohort has 7,882 Source 1 entities and excludes name
groups used in the earlier DARPA development sample. Its 7,909-entity reserve
cohort was evaluated once after the owner/context policy was frozen. Local tune
and reserve scores, the earlier audit, and Portal scores measure different
populations.

| Approach | Same-cohort result | Outcome |
| --- | ---: | --- |
| Frozen CatBoost baseline | 0.9775204 macro F0.5 on new tune; historical audit 0.9737442; user-reported Portal 0.9608 | Validated fallback archive retained unchanged |
| 300-step Granite embedding pilot, one positive target per owner | True-owner retrieval recall 0.9895455 at top 10 of 27,261 tune links | Below predefined 0.99 retrieval gate; no full-target pilot inference |
| HN1: 100,000 positive and 100,000 label-confirmed wrong-owner pairs; 12,217 negatives have the same normalized name | True-owner retrieval recall 0.9915630 at top 10; 55 more links than pilot | Full train target retrieval completed; not an F0.5 score |
| One-million-pair full Granite encoder | True-owner retrieval recall 0.9953780 at top 10; 104 more links than HN1 | Chosen for first full owner model; retrieval is not F0.5 |
| Full encoder + pair LightGBM + incoming-context LightGBM | 0.9829591 macro F0.5 on same tune; +0.0054387 versus frozen baseline, paired name-group bootstrap interval [+0.0034172,+0.0076403] | Current best measured challenger. Threshold 0.62 was tuned on this cohort, so independent performance remains unmeasured |
| SEL1: second rival-aware LightGBM chooses among ten owners and can reject all | 0.9798411 macro F0.5 on same tune; +0.0023207 versus frozen baseline but 0.0031180 below current challenger | Not promoted. Wrong-owner rescue versus regression analysis may inform a narrower combined policy |
| Qwen3-Reranker-0.6B LoRA on 200,000 mined pairs, selective reranking of all routed tune groups | Best tested override policy: 0.9741627 macro F0.5 on the same tune cohort; 26,573 TP, 525 FP, 688 FN | Not promoted: 0.0087965 below the owner/context model and 0.0033578 below the frozen baseline. The best sweep point used Qwen score ≥0.999 without a margin or rejection requirement; false positives increased and singleton accuracy fell to 0.91646. See [paired report](reports/qwen06_paired_tune.json) and [sweep](reports/qwen06_policy_sweep.json). |

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

The Slurm bounded scoring smoke hit its account NFS quota after 512 India groups.
The same model and inputs were transferred with exact hashes to A6000; a bounded
smoke completed there before all 6,110 India and 7,447 US routed tune groups
were scored. The Qwen sweep reproduced the owner/context base score before
applying overrides, so this regression is not a base-score mismatch.

The owner/context policy was selected on tune and frozen at threshold 0.62
before one reserve evaluation. On 7,909 reserve entities it reached
**0.9840808** macro F0.5 versus **0.9807521** for the frozen CatBoost baseline
on those same entities, a paired gain of **+0.0033287**. The 2,000-replicate
name-group interval is **[+0.0013414, +0.0052273]**. See
`reports/reserve_paired_f05.json`. This independent improvement supports a
separate full test submission; complete test inference and validation are the
remaining gates. Test labels and Portal accuracy remain unknown.
