# DARPA dense-owner experiment

This is a separate experiment. The frozen CatBoost submission and its ZIP remain
the reproducible fallback. On 27 September 2026 the user reported DARPA's Portal
macro F0.5 as **0.9608**. Their screenshot shows Banana at **0.990816**. The local
CatBoost audit is 0.9737442188439884 on a different population.
Measured variants and failures are recorded in [EXPERIMENTS.md](EXPERIMENTS.md).

The architecture follows the target-to-Source-1 retrieval idea documented in
`SERVER_APPROACH_REVIEW.md`: contrastively train a multilingual encoder using only
the supplied records, retrieve a small owner shortlist, and fit a precision-aware
matcher using owner competition. This implementation was written independently;
the other server experiments' predictions and trained weights are not substituted
into the DARPA archive.

The base model is [IBM Granite embedding 97M multilingual r2](https://huggingface.co/ibm-granite/granite-embedding-97m-multilingual-r2),
Apache-2.0, pinned to `835ad14087e140460703cf0fae09f97d469d65c2`.
It uses CLS pooling and normalized 384-dimensional vectors. All fine-tuning data
comes from the challenge training TSVs. Model downloads are separate from offline
training and inference; no business-identity lookup is used.

## Validation design

`prepare.py` normalizes Source 1 names and partitions whole country/name groups
with a deterministic hash. It reserves approximately 0.5% each for tuning and a
sealed reserve, 1.5% each for pair fitting and context fitting, and the rest for
encoder fitting. All groups represented in the earlier 122,000-entity DARPA
development sample are excluded from the new evaluation splits. One positive
target per fitting owner trains the encoder; different owners with identical
record text or the same normalized name are masked as contrastive negatives.

The prepared split contains 23,770 pair-fitting entities, 23,746 context-fitting
entities, 7,882 tuning entities, and **7,909 sealed reserve entities**. There are
100,492 historical normalized-name groups excluded from the new evaluation.
The first full encoder run used one million positive fitting pairs.

Excluding historical groups can change the name-frequency mix. Any performance
comparison must score the frozen baseline on the same new entities and report
country/name-frequency strata; a fresh local score is not a Portal estimate.

The first probe measures **candidate recall only**, using all Source 1 records as
possible owners and tune positive links as queries. It does not estimate F0.5 or
singleton accuracy, because that requires processing all target records and
counting incoming false positives. The reserve is not opened by this probe.
Candidate counts and top-k recall must be reported together under the updated
competition rules. Never shorten a candidate file after scoring and claim it was
the candidate set fed to the model.

## Current stages

1. Prepare full supplied records and split manifest.
2. Smoke-test cached contrastive gradients and a real GPU training step.
3. Train a short A6000 pilot and a full one-million-pair A100 run.
4. Encode all training Source 1 records; measure exact owner recall at k=1,2,4,8,10,20,32.
5. After retrieval is measured, fit owner and incoming-link decision models using
   full-target retrieval, then evaluate once on the sealed reserve.
6. Fine-tune Qwen3-Reranker-0.6B on fitting-role positive pairs and retrieved hard
   negatives. Score complete shortlisted owner groups for uncertain targets and
   calibrate final decisions using its evidence. This is the primary challenger
   architecture requested by the user; promotion still requires measured gains.

Only successful held-out improvement and complete validators can promote a new
pipeline to a replacement submission. Training, retrieval, and official Portal
scores must be distinguished in every report.

The public repositories supplied later are reviewed, with revision links and
evidence limitations, in [PUBLIC_APPROACH_REVIEW.md](PUBLIC_APPROACH_REVIEW.md).
The strongest server lead remains the contextual owner model plus selective Qwen
reranking. Its reported development score does not validate our implementation.

The requested upgrade order is **HN1** (mined hard-negative continuation),
**SEL1** (owner selection with rejection), **Q4B1** (bounded 4B comparison), then
**MV1** (multiple positive views if retrieval misses remain). Research sources and
their limitations are in `research/ONLINE_APPROACH_UPGRADES_20260927.md` at the
repository root. Compare identical tune entities,
candidate budgets and routing masks wherever the ablation permits; no benchmark
paper supplies a DARPA score forecast.

`mine_hard_negatives.py` produced 100,000 positive and 100,000 label-confirmed
wrong-owner pairs from fitting-role records. Of the wrong-owner pairs, 12,217
have the same normalized name. `train_mined_encoder.py` implements a separate
HN1 continuation from the pilot checkpoint, consuming these pairs. It permits distinct
same-name negatives while retaining same-owner and indistinguishable-text masks.
The rectangular gradient-cache case is covered by the GPU parity test. HN1
training and the same-tune retrieval probe completed. The one-million-pair full
encoder retrieved more true owners at the same candidate budget, so it is used
for the complete owner/context experiment. SEL1 was measured separately;
Q4B1 and MV1 remain optional follow-ups.

`baseline_probe.cpp` reuses the frozen predictor's functions to evaluate selected
queries against the full record pools. `cluster/baseline.sh` first checks exact
candidate and match parity on eight existing test rows, then predicts the fresh
tune cohort. `retrieve_owners.py` performs resumable exhaustive same-country
retrieval on every target; a `--max-rows` smoke run is explicitly incomplete.

`build_features.py` keeps all competing owners for any selected target.
`fit_owner.py` fits the pair model on pair-role targets, then an incoming-link
model on context-role winners. Held-out targets are excluded from each fitting
stage. Complete incoming context is used only for selected owners; aggregates
for partially observed rival owners are never fitted or evaluated. Equal-score
incoming ranks are tied, so target-ID ordering cannot break probability ties.
The context model's threshold is selected on tune entities only.

`prepare_qwen.py` uses only encoder-role positives and encoder-role hard-negative
owners, leaving the pair/context/tune/reserve owners out of reranker fitting.
`train_qwen.py` fits a rank-16 LoRA adapter; `qwen_model.py` scores yes/no logits
with a fixed business-matching instruction. The pinned Qwen base revision is
`2925c98b11f00b3364acaeb0a669f498ac45bf54`, Apache-2.0. Selective Qwen
reranking was tested on the full routed tune cohort and regressed; it is not the
selected production policy.
`score_qwen_groups.py` records exact retrieved owner identities and scores every
owner in a routed group; its current routing uses only pair model probabilities,
not labels. A bounded comparison may reuse exactly those groups. The scored
tune-group identities and decision integration passed validation.
The first 1,000-group-per-country scoring smoke stopped after 512 India groups
because the Slurm account reached its NFS disk quota while another project was
also using that account. This was an environment failure, not a model outcome.
The complete model, retrieval, feature, adapter and Qwen-base inputs were moved
to the A6000 workspace with matching file hashes. The bounded scoring smoke
completed there. Selective scoring then covered all 6,110 India and 7,447 US
routed tune groups. The best tested Qwen override policy scored 0.9741626707
macro F0.5, below both the owner/context model (0.9829591299) and the frozen
baseline (0.9775204401). Its 525 false-positive links and 0.91646 singleton
accuracy make it unsuitable for promotion. The owner/context policy was selected
and then evaluated on reserve. See `reports/qwen06_paired_tune.json`.

`fit_selector.py` adds the requested **SEL1** alternative: a second LightGBM
learns from whole shortlisted owner groups after the pair model, chooses among
all ten owners, and rejects a group when its best probability falls below a
tuned no-match threshold. It fits on the separate context role and evaluates on
the tune role. A synthetic end-to-end role/identity check passed; challenge
performance measured **0.9798410958066807** macro F0.5 on the same tune cohort.
This is above the frozen baseline but below the current pair/context challenger
by 0.0031180, so SEL1 is not selected as a replacement policy.

## Measured status

The frozen baseline scores **0.9775204400514894** on the new 7,882-entity tune
population, with 25,913 true-positive links, 145 false-positive links and 1,348
missed links. This uses every training target and competing Source 1. Its eight
test-row subset check matched both frozen TSVs exactly. Reports are in `reports/`.
The first complete challenger (full encoder, pair LightGBM, incoming-context
LightGBM, no Qwen) scores **0.9829591299071904** on the same 7,882 tune
entities, versus **0.9775204400514894** for the frozen baseline. It has
26,506 true-positive, 206 false-positive and 755 missed links. The paired
difference is +0.0054387; a 2,000-replicate name-group bootstrap interval is
[+0.0034172, +0.0076403]. Country and name-frequency breakdowns are in
`reports/full_owner_paired_tune.json`. The 0.62 decision threshold was selected
on this tune cohort, so the resulting tune F0.5 is optimistic for generalization.
The selected policy was evaluated once on the sealed 7,909-entity reserve after
freeze. It scored **0.9840807683** macro F0.5 against **0.9807520537** for
the frozen CatBoost baseline on the same entities, with a paired name-group
bootstrap interval for the gain of **[+0.0013414, +0.0052273]**. The threshold
was not adjusted after reserve evaluation. This is not a Portal or test score;
the Portal score for this challenger is unknown. SEL1 and Qwen were compared on
tune and did not improve the selected policy.
The independent frozen inference entry point, `predict_frozen.py`, reproduced
all 7,882 tune matching rows exactly from the saved models and complete feature
graph. Its candidate file contained 333,635 actual scored pairs and recovered
27,135 of 27,261 true links; the streaming evaluator reproduced the same
0.9829591299071904 macro F0.5. See `reports/predict_frozen_parity.json`.
The optional reranker-routing export was also checked against the original
fit-stage winner graph: target and owner rows match exactly across 1,957,017
groups, pair probabilities differ only by float32 rounding (at most 5e-8), and
context probabilities for the 258,104 fitted/evaluated winner rows agree within
1e-12. Its matching and candidate TSVs are byte-identical to the first frozen
inference run.

The pilot's same-population positive-link retrieval recall is **0.9895455 at
k=10**, **0.9921500 at k=20**, and **0.9934338 at k=32**; top-1 is 0.9682330.
The HN1 continuation reaches **0.9915630 at k=10**, **0.9937273 at k=20**,
and **0.9947544 at k=32** on exactly the same 27,261 tune positive links.
Thus HN1 retrieves 55 additional true owners at k=10. This is a retrieval
result, not a challenge F0.5 score; the full target pool still needs owner
classification and incoming false-positive accounting. The detailed HN1 report
is `reports/hn1_retrieval_probe.json`.
The completed one-million-pair full encoder reaches **0.9953780 at k=10**,
**0.9969554 at k=20**, and **0.9978357 at k=32** on the same links. It retrieves
104 more true owners at k=10 than HN1. Its report is
`reports/full_retrieval_probe.json`. We are processing the full train target
pool with both runs. The full encoder was selected for the complete pair/context
model because its same-tune retrieval recall was higher. Full retrieval completed
for every training target (4,133,346 India; 6,186,873 US), and complete-rival
features were built for 655,543 India and 1,301,474 US target groups. The feature
files include all ten retrieved owners for each selected group; 19,570,170 pair
rows in total. Every target group was scanned; feature extraction omitted only
groups whose ten owners were all outside the pair, context, and tune roles.
Pair/context model fitting and exact paired tune F0.5 evaluation completed on
Slurm; results appear above. Test retrieval and feature extraction remain separate.
The frozen baseline retrieves 0.9876013 of true links with 160 targets per Source
1 on these tune entities. These have different candidate directions and budgets;
the challenger needs all-target processing before its exact total budget is known.
The pilot top-10 result did not clear the predeclared 99% processing gate, so the
expensive control pipeline stopped before full-target inference. HN1 cleared
the gate; its 1,024-target full-pool smoke passed finite-score, score-order,
unique-owner, and owner-bound checks. Its complete train target retrieval also
finished on the A6000 server. Reserve labels were opened only for the single
post-freeze evaluation described above.

The Qwen 0.6B adapter training smoke completed two optimizer steps on 16 supplied
training pairs. It is only an execution check, not a fitted competition model.
The real Qwen adapter finished 3,125 optimizer steps over 200,000 mined fitting
pairs (100,000 positive and 100,000 confirmed wrong-owner). Its completion
manifest and adapter are retained locally and on the servers. A bounded run
scored 1,000 routed owner groups per country on A6000 to test inference cost and
group identity checks before selective use. This training completion
does not establish a challenge F0.5 gain.

The 300-step encoder pilot is trained; its local checkpoint SHA-256 is
`af830261e86816d5a1c6df4238d6c825deabbd5f1bd9cd2972d160bb4897bca9`.
Saved-tokenizer parity passed on 1,000 supplied fitting records despite a
Mistral-regex warning emitted by Transformers while loading the local Granite
tokenizer. No tokenizer regex was changed. Gradient replay, numeric and
Unicode features, leave-one-edge-out support, probability ties, and the synthetic
end-to-end owner pipeline have been checked. Synthetic scores are not challenge
accuracy results. Full-size feature extraction and Qwen training completed;
their effect on independent holdout performance remains to be tested.

## Execution

The A6000 environment uses `requirements.txt`. Slurm uses the cluster's
`pytorch/2.10` module (Python 3.10, NumPy 2.2.6) with the same Transformers 4.57.6,
PyArrow 23.0.0, DuckDB 1.4.1, LightGBM 4.6.0, RapidFuzz 3.14.1, and FAISS 1.12.0.
Each scheduled run writes its complete `pip freeze` to its log directory.
Training, encoding, and probing load downloaded local weights in offline mode.

Run `src/prepare.py --help` to prepare the data. The GPU gradient-cache test checks
both ordinary gradients and dropout RNG replay against retained-graph backprop.
`cluster/pilot.sh` runs this check, a real two-step training smoke, a 300-step pilot,
and its full-owner-pool retrieval probe. Submit `cluster/train.sbatch` from the
experiment root after preparing `work/data` and `work/base_model`; it performs
the full encoder fit and the same retrieval probe using one A100.

From this experiment directory, with the supplied dataset available externally:

```sh
python src/fetch_base.py work/base_model
python src/prepare.py --dataset /path/to/dataset --output work/data \
  --previous-ids splits/previous_development_ids.txt --max-encoder-pairs 1000000
bash cluster/pilot.sh       # a local CUDA host, GPU 0
# Or, after preparing the Slurm environment and creating logs/:
sbatch cluster/train.sbatch
```

The historical ID list contains only Source 1 identifiers from supplied training
data; it makes the fresh split exclusions reproducible. Code is MIT licensed;
the encoder base and derivative weights retain their Apache-2.0 provenance.

## Reproduce the selected final output from the archive

The submission ZIP places these Python files directly under
`code/business_entity_resolution/src/`, the frozen encoder under `models/encoder/`,
the pair/context models under `models/owner/`, and the historical exclusion IDs
under `splits/`. Use Linux with a CUDA GPU and the pinned requirements. The
challenge's original `dataset/` directory should sit next to `code/` at the
extracted archive root. From `code/business_entity_resolution/`, run:

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python src/prepare.py --dataset ../../dataset --output work/data \
  --previous-ids splits/previous_development_ids.txt
.venv/bin/python src/encode.py --data work/data --model models/encoder \
  --output work/test_vectors --glob 'test_s1_*.parquet' --batch-size 512
.venv/bin/python src/retrieve_owners.py --data work/data \
  --vectors work/test_vectors --model models/encoder \
  --output work/test_retrieval --split test --batch-size 512 --k 10
.venv/bin/python src/build_features.py --data work/data \
  --retrieval work/test_retrieval --output work/test_features \
  --split test --workers 6
.venv/bin/python src/predict_frozen.py --data work/data \
  --features work/test_features --retrieval work/test_retrieval \
  --model models/owner --threshold 0.62 --split test \
  --source1-tsv ../../dataset/test/test_source1.tsv --output work/reproduced_output
```

The final command writes both required TSVs and `run.json`. Retrieval preserves
ten candidates per target, feature extraction scores all ten, and the candidate
TSV reverses exactly those scored edges per Source 1 entity. These commands
require space for complete test vectors, retrieval arrays, pair features, and
TSVs; the provided `cluster/` scripts show the staged server execution used
for this run. `run.json` records file and model hashes so a reproduction can be
compared with the packaged output.
