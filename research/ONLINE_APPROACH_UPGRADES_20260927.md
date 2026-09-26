# Online approaches to test beyond the current DARPA plan

Search date: 27 September 2026. Searched challenge-specific GitHub/Hugging Face results, published entity-matching research, recent preprints, author repositories and official model cards. No inspected public result verifies that a released solution beats the server reference's claimed 98.7% Portal score or reproduces Banana's 99.0816%. The recommendations below are experiments, not measured DARPA improvements.

## 1. Train on real confusing alternatives

**Evidence:** HELEA (May 2026 preprint) constructs same-name but distinct-entity negatives and demonstrates the weakness of name-dependent matching on those examples. Its input includes knowledge-graph context, so its F1 results do not transfer directly to business-name/address matching. The transferable idea is to make training negatives resemble the alternatives encountered at inference. [Paper](https://arxiv.org/abs/2605.28308).

NV-Retriever studies positive-aware filtering during hard-negative mining to avoid treating plausible positives as negatives. It evaluates general retrieval, rather than this competition. [Paper](https://arxiv.org/abs/2407.15831).

**Local finding:** `student_resource/code/darpa_dense/src/train_encoder.py` lines 140–142 currently masks every same-normalized-name owner pair from contrastive negatives. `prepare.py` chooses one positive target per fitting owner. This is a defensible initial safeguard, but means the pilot does not explicitly learn from many of the hardest same-name owner alternatives or multiple noisy views of one owner.

**DARPA adaptation:** after the current encoder finishes, retrieve top alternatives for fitting targets only. Add label-confirmed wrong owners with confusing names, similar streets, nearby house numbers, and transliteration similarity. Preserve masks for indistinguishable serialized texts and uncertain cases. Include true examples with number shifts so the model cannot learn a universal number-mismatch rejection shortcut. Sample multiple available positive views across training rounds, respecting all existing fit/tune/reserve boundaries.

Run this as a separate checkpoint and compare top-1/top-10 recall and eventual macro F0.5. Removing the same-name mask alone is not equivalent to a validated hard-negative curriculum.

## 2. Select an owner from the shortlist, including no match

**Evidence:** COMEM, COLING 2025, studies matching, comparing and selecting, and combines inexpensive filtering with selection among a smaller candidate set. It explicitly provides a none-of-the-above option and reports candidate-order bias. Its benchmarks are much smaller than our full workload. [Paper](https://aclanthology.org/2025.coling-main.8/), [author code](https://github.com/tshu-w/ComEM).

Mistral4SelectEM trains for selective matching with contrastive margin ranking, targeting semantically similar hard negatives rather than only independent pair classification. Its published product benchmarks are not Amazon ML Challenge 2026. [Paper](https://doi.org/10.1016/j.aei.2025.103538), [author code](https://github.com/quickhdsdc/LLM4EntityMatching).

**DARPA adaptation:** make the inference unit a target record and its shortlisted Source 1 owners. Train a grouped scorer to choose an owner or reject the group. Include unmatched targets and examples whose true owner is absent from the shortlist. A Source 1 may receive several targets; do not impose one-to-one matching on Source 1. Calibrate rejection and final decisions against macro F0.5.

The server A9 mainly accepts/rejects its existing owner winner. Its saved development error budget has 761 wrong-owner true links and 386 retrieval misses. Selection can address an error category that winner-only acceptance cannot repair. Those counts belong to A9's cohort, not DARPA's audit. No gain is established until the new scorer wins on our shared holdout.

A practical first version is a learned group scorer over pair, owner and Qwen features. A generative listwise model is a subsequent alternative; randomize candidate order during training and test order sensitivity.

## 3. Try Qwen3-Reranker-4B on a bounded difficult subset

**Evidence:** Qwen's official model card reports MTEB-R 69.76 for 4B versus 65.80 for 0.6B, and multilingual MMTEB-R 72.74 versus 66.36. Both comparisons use the model-card evaluation setup, with candidates from Qwen3-Embedding-0.6B. These are retrieval metrics, not business entity matching or F0.5. The 4B model is Apache-2.0. [Official model card](https://huggingface.co/Qwen/Qwen3-Reranker-4B).

**DARPA adaptation:** benchmark it on the exact uncertain candidate groups scored by the smaller model. Compare zero-shot and challenge-pair fine-tuned variants where time permits; a generic larger reranker may lose to a specialized smaller one. Add its scores to the decision model or grouped selector, rather than accepting its output without calibration.

Use the A100 allocation when available, or profile a bounded A6000 run. Measure pairs/second, memory, routed group count and projected full-test runtime before choosing the routing fraction. The pre-existing server union contains about three million target groups; even selective routing can remain expensive. Choose a narrower route using training/tuning evidence if necessary, and preserve complete shortlisted groups.

## 4. Train retrieval for multiple views and measure candidate efficiency

**Evidence:** SC-Block uses supervised contrastive learning to cluster matching records and construct small candidate sets. The ESWC 2024 version reports candidate sets about 50% smaller on average and complete pipelines 1.5–4 times faster on its benchmarks without F1 loss. These are product-matching results, not a guaranteed speedup for DARPA. [Published paper](https://2024.eswc-conferences.org/wp-content/uploads/2024/04/146640116.pdf), [author code](https://github.com/wbsg-uni-mannheim/SC-Block).

**DARPA adaptation:** use multiple supplied S2/S3 views per training identity, rather than always the same one. Ensure multiple positives are never treated as negatives. Compare name-only, address-only and combined retrieval channels for incremental recovery at equal total candidate budgets. This extends the current dense experiment rather than requiring an unrelated model replacement.

## Recommended experiment order

1. Keep the current pilot as the control. Complete the same-entity frozen-baseline comparison.
2. **HN1:** one continuation trained with mined, label-confirmed hard negatives. Keep architecture, candidate budget and evaluation fixed.
3. **SEL1:** a grouped owner selector with rejection versus the winner-only decision layer, using identical candidate sets and available scores.
4. **Q4B1:** replace or augment only the routed reranker scores; first hold routing and candidates fixed so the comparison isolates the model change.
5. **MV1:** multiple-positive/multi-view retrieval if retrieval misses remain material. Measure recall and actual candidates together.

Select changes on tuning data and evaluate the chosen pipeline once on the sealed reserve. Preserve all target distractors, source competitors, empty entities and unretrieved truth in end-to-end scoring. Benchmark scores from these papers are not forecasts of Portal gains. This research task did not alter training code, active jobs, model weights or submissions.
