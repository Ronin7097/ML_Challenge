# DARPA inference and reproduction

The packaged checkpoint and frozen policy reproduce the final rich-context
pipeline. All inference code uses local model files and supplied competition
records. Read `../../Documentation_template.md` for methodology and limitations.

## Environment

Use Linux, Python 3.13, a CUDA GPU supporting bfloat16, at least 48 GB host RAM,
and sufficient scratch disk for embeddings, candidate features and output.
The full official validator materializes candidate sets, so it is run on a
large-memory machine. Input TSVs must be read with a tab separator.

Put the official `dataset/train/` and `dataset/test/` beside `code/` in the
submission root. The packaged `models/` contains the exact producing encoder,
pair model, five base-context models, five rich-context models and frozen policy.
`models/manifest.json` lists their hashes. Git excludes the weight files; obtain
them from the local final submission ZIP when using a fresh GitHub clone.

Run from `code/business_entity_resolution/`:

```sh
python3.13 -m venv .venv
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
  --split test --workers 8
.venv/bin/python src/predict_frozen.py --data work/data \
  --features work/test_features --retrieval work/test_retrieval \
  --model models/owner --threshold .62 --split test \
  --source1-tsv ../../dataset/test/test_source1.tsv \
  --output work/base_output --routing-output work/base_output/routing
.venv/bin/pip install -r requirements-rich.txt
.venv/bin/python src/predict_rich.py --data work/data \
  --features work/test_features --retrieval work/test_retrieval \
  --base-output work/base_output --base-model models/owner --model models/rich \
  --policy models/rich/policy.json --output work/rich_output --workers 8
```

The base stage used NumPy 2.4.1; the rich stage used NumPy 2.2.6. Install the
second requirements file only after base inference. The latter is an overlay
on the base environment. Completed work directories are protected from reuse
with different inputs or settings; start with fresh work directories for a new
run. Float16 GPU retrieval can have hardware-dependent ties; preserve the pinned
runtime and inspect parity before treating a new run as byte-identical.

## Export and validate

The following commands replace the root output copies with the regenerated
ones, so preserve any earlier output you need before running them:

```sh
mkdir -p ../../output
cp work/rich_output/matching_results.tsv ../../output/matching_results.tsv
cp -L work/rich_output/candidate_pairs.tsv ../../output/candidate_pairs.tsv
.venv/bin/python src/validate_submission_official.py \
  --matching ../../output/matching_results.tsv \
  --candidate ../../output/candidate_pairs.tsv \
  --test-dir ../../dataset/test --check-ids
.venv/bin/python src/audit_submission.py \
  --test-dir ../../dataset/test --output ../../output \
  --report work/regenerated_validation.json
```

`src/validate_submission_official.py` is the unmodified supplied validator.
`validation.json` records the full completed audit for the delivered TSVs;
`run.json` records their producing model, policy, source and dependency hashes.
A new inference run has its own `work/rich_output/run.json`; retain it with its
new audit rather than mislabeling regenerated files with the delivered run's
metadata. The package checker intentionally accepts only the delivered frozen
identities until the associated manifests are deliberately updated.

## Checks and supporting source

```sh
.venv/bin/python -m unittest discover -s src -p 'test_*.py'
```

Tests cover pair features, macro F0.5/singletons, rich context and exact rival
probability recovery. The gradient-cache test needs CUDA and skips on a CPU.
The `validation/` directory contains local score reports and exact cache-parity
checks. `prepare.py`, `train_encoder.py`, `fit_owner.py`, `build_rich_context.py`,
`fit_rich_context.py` and `rich_policy.py` retain the training implementation;
training is not required to reproduce predictions with packaged checkpoints.

The code discovers country partitions from supplied records, including France.
All ten candidate edges per target enter the pair matcher; later owner selection
and rich acceptance are matching stages. Every predicted ID remains in the
candidate file. No alternative experimental model is used in this submission.
