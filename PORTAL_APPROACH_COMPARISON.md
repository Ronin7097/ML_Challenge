# Approach comparison and DARPA improvement plan

Evidence checked on 27 September 2026, around 02:30–02:45 IST. This review combines the supplied research note, the previous chat **Improve F0.5 to 98%**, public repositories, and read-only inspection of both supplied servers. Credentials are not stored here. Training in the previous chat is still active; this review does not launch duplicate jobs or replace its files.

## Recommendation

The strongest measured server architecture found is **Granite multilingual target-to-owner retrieval → LightGBM pair scoring → A8 owner/incoming-link context → selective Qwen reranking → final accept/reject decision**. Reproduce and evaluate that architecture within DARPA's new split. Preserve lexical/structural retrieval as a complementary channel if it recovers additional true links at a reasonable candidate budget.

This is the best-supported direction among the inspected approaches, not a verified reconstruction of the leaderboard leader. No inspected public repository establishes that it produced Banana's score.

## What the scores actually mean

| Approach | Score | Evidence and limitations |
|---|---:|---|
| DARPA current Portal | **96.08%** | User-reported in the previous chat. |
| Banana leader screenshot | **99.0816%** | Recorded in the previous chat; not independently rechecked against the live leaderboard. Gap to DARPA: 3.0016 percentage points. |
| DARPA frozen CatBoost | **97.3744%** | Independent 6,000-entity historical audit. Different population from Portal. |
| Server `.155` dense owner | **98.0542%** | Development on 19,991 entities. Earlier notes report 97.4% public, without independent Portal verification. |
| Server `.155` A8 context | **98.5913%** | Grouped development on the same 19,991 entities. Release notes refer to an A8 file as public 97.7%. |
| Server `.155` A9 Qwen | **99.1306%** | Same development cohort; repeatedly explored. Release notes associate the complete output with public **98.7%**, but that is a server-recorded claim, not independently verified Portal evidence. |
| Server `.132` E012 | **97.7058%** | Full-density development on 20,071 entities; 1,769,361 fitting anchors. Reported paired gain over its own baseline: 1.1876 points, interval 1.0722–1.3078. Not a head-to-head comparison with DARPA. |

A9's complete output is at `/home/jagat/claude-merged-20260926/work/sub_a9_full/submission/`. Its matching SHA-256 is `c9fdeca0c2098d7883987a16750d865b5eb608b8df40d678629c8e5941deeaab`. Existing validator logs report 1,732,544 rows, successful candidate/subset checks, and successful matching-ID existence checks. These logs establish formatting, not accuracy. The coverage report has zero uncertain-union winners waiting for Qwen in all three countries.

**Update to the September 26 review:** full A9 test output now exists. Its release notes now contain a 98.7% public-score reference. Neither fact turns the 99.13% development result into a Portal score.

On `.132`, `artifacts/E014/results.json` still identifies itself as E012 and carries the same E012 results. Its presence is not evidence that the E014 variant has a new independently measured score. E014 inference jobs were running at inspection; their completion and score remain unverified.

## Corrections to the pasted public-repository summary

- **AyanAhmedKhan:** useful implemented ideas include five sparse views, reverse target-to-S1 retrieval, and out-of-fold collective context. The repository explicitly warns that its 98.703% stacked result used fold-restricted competition features; the gain may not transfer to test. Use the architectural ideas with full-graph context. [Repository](https://github.com/AyanAhmedKhan/amazon-ml-challenge), [validation caveat](https://raw.githubusercontent.com/AyanAhmedKhan/amazon-ml-challenge/main/PROJECT_STATE.md).
- **LearnerSanket24:** the README labels both the performance table and ensemble ablations **Expected**. Those numbers are proposals, not demonstrated improvements. They do not justify replacing DARPA with an XGBoost/LightGBM/CatBoost blend. [README](https://github.com/LearnerSanket24/AmazonMLChallenge#-expected-performance).
- **Akash-bardia:** reports 97.6105% validation with lexical blocking and LightGBM. Its cohort differs from ours, so the number does not establish an improvement over DARPA. [Repository](https://github.com/Akash-bardia/amazon-ml-challenge-2026).
- **chaXin44:** the README primarily describes the challenge and noise patterns; it does not establish a top-scoring model. [Repository](https://github.com/chaXin44/amazon-ml-challenge-2026).

The transferable public insight is to calculate competing-owner evidence across the full graph and learn sibling support from out-of-fold scores. Ayan's feature catalog describes both target-side margins and within-owner ranks; these should be computed before selecting the evaluation entities. [Feature catalog](https://raw.githubusercontent.com/AyanAhmedKhan/amazon-ml-challenge/main/FEATURE_CATALOG.md).

## How this relates to our existing model

DARPA already has normalization, country blocking, approximate transliteration, 107 pair features, a 156-feature context model, competing-reference evidence, and expected-F0.5 set selection. Reimplementing those basics or adding a third tree library is unlikely to address the largest measured gaps.

From the already-published audit aggregates, without reopening labels:

| Failure stage | True/false links |
|---|---:|
| True links absent from the candidate set | **522** |
| True links retrieved but rejected | **801** |
| False links accepted | **112** |

There are 20,764 true audit links. Candidate recall is 97.4860%; even perfect classification on those candidates reaches only 99.1583% macro F0.5. The decision gap to that ceiling is 1.7839 percentage points. These counts identify useful work; they are not additive estimates of achievable score gains.

| Priority | Concrete change | Reason and comparison |
|---|---|---|
| 1 | Finish the independent dense owner retriever; test lexical union at fixed budgets | Target the 522 retrieval misses. Report recall at k=1,2,4,8,10,20,32, plus total/mean/p95 candidate counts and incremental recall after the deployed cap. |
| 2 | Fit owner competition and leave-one-target-out incoming support | Target the 801 rejected true links and hard false matches. Compare against the frozen CatBoost on exactly the same new entities, with all target distractors and Source 1 competitors present. |
| 3 | Add a Qwen reranker score for uncertain complete owner groups | A9 improves A8 by 0.5392 development points on its own cohort. Train and calibrate within DARPA's split; do not transfer its threshold blindly. |
| 4 | Test a learned reranker that can change the proposed owner | A9 primarily accepts/rejects the existing owner winner. Its own error budget contains 761 wrong-owner true links versus 386 retrieval misses. A learned owner selection stage could address this, but remains a hypothesis. |
| 5 | Test country transfer and density robustness | France has no supplied labels. Use held-out-country proxies and country/name-frequency breakdowns, without presenting a proxy as France accuracy. |

The prior tuning test found no duplicate selected target owners and no gain from a score-only singleton veto. Repeating those filters on the same small cohort is lower priority than full-pool competition.

## Specific architecture details worth reproducing

**Encoder reference configuration:** Granite embedding 97M multilingual r2, pinned revision `835ad14087e140460703cf0fae09f97d469d65c2`; CLS pooling and normalized vectors; one epoch on 1,323,827 target/owner positive pairs; batch 1,024, microbatch 128, maximum length 96, learning rate 7e-5, cached multiple-negative loss with temperature 0.05. The current DARPA pilot uses its own fitting pairs and separate holdouts.

**A8 context:** winner probability, runner-up probability, margin, number of plausible owners, incoming candidate count, other targets won by the owner, other strong S2/S3 support, accepted-link support, lost strong claims, other-winner probability sum, and within-owner rank. Subtract the current edge from support features. Compute these aggregates on the complete unlabelled candidate graph, using out-of-sample model scores where required.

**A8 learner:** 204 features; LightGBM binary objective, 400 rounds, 31 leaves, learning rate 0.05, minimum leaf size 100, feature fraction 0.8, L2 5. A9 adds ten Qwen group features, producing 214 features, and averages five seeded models with 80% row sampling. Its development-selected threshold is 0.84. These are reference settings to validate, not a guaranteed DARPA configuration.

**Qwen group features:** winner score, strongest alternative score, winner margin, whether Qwen prefers the existing winner, winner rank, original runner-up score, group maximum, second-best score, count above 0.5, and score sum. Score the same complete shortlisted owner group in training and inference. Store missing scores explicitly and reject incomplete scored groups or mismatched pair identities.

**Avoid universal house-number vetoes.** The `.155` France-format transfer diagnostic cut 94 links, all labelled true in its US/India diagnostic, reducing macro F0.5 by 0.0597 points. Number edits, shifts, unit numbers, and peer agreement should be features. The original France detector had zero supported locality transfers, so this is evidence against a universal rule, not an estimate of France performance.

**Do not assume more bagging helps.** The server's ten-bag variant scored 99.1207%, below the five-bag A9 result. Its fixed rejected-owner rescue produced zero rescues and failed its promotion gate. These are useful negative results.

## Execution state and next acceptance test

- **172.24.16.155:** three RTX A6000 48 GB GPUs. DARPA's 300-step pilot has completed training and is encoding Source 1 records under `/home/jagat/.DARPA_submission_work/dense_owner/`. Other GPU workloads are present.
- **172.24.16.132:** Slurm cluster with A100 resources. DARPA job **12232**, `DARPA-granite`, requests one GPU, 8 CPUs, 24 GB RAM and eight hours. It was pending with `AssocGrpCpuLimit`; available physical GPUs do not bypass the shared account limit.
- The previous chat is implementing full-pool retrieval and the frozen-baseline comparison in the shared `student_resource/code/darpa_dense/` folder. This review leaves that work and both servers' existing experiments intact.

Use the existing 23,770 pair-fit, 23,746 context-fit, 7,882 tuning and 7,909 sealed-reserve entity splits. The split excludes historical development name groups. First compare baseline, dense owner, owner context, and selective reranking on the same tuning population. Select the configuration before opening the sealed reserve, then report paired macro F0.5, precision/recall, singleton errors, candidate budget, country and name-frequency strata. A positive-pair retrieval probe alone cannot establish end-to-end F0.5.

Before promoting a replacement, generate complete outputs, validate all IDs and candidate subsets, retain the actual pre-decision candidate set, and package the producing source/model/configuration together. The review itself changes no trained weights and establishes no new DARPA Portal score.

## Saved evidence

`research/server_comparison_20260927/` contains snapshots of A9 development/coverage/error-budget reports, the Granite training configuration, E012 and E014 reports, and a manifest with source paths, capture time and SHA-256 hashes. These are small aggregate reports, not model weights, predictions or business records. The original report's exploratory-development limitations still apply.
