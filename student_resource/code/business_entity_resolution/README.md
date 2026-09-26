# Business entity resolution pipeline

The improved pipeline uses a CatBoost pair scorer, candidate context, and an unlabelled index of all Source 1 records to compare competing matches. A decision rule chooses how many candidates to return for each entity by approximating expected F₀.₅, including the empty-set option. It was selected on a separate tuning split. A further audit split is excluded from fitting, early stopping, error inspection, and decision selection.

See `EXPERIMENTS.md` for the full development history, discarded variants, and suggested improvements.

## Inference

From the extracted DARPA archive root, place the supplied `dataset/` directory alongside `code/`, then run:

```sh
python3 code/business_entity_resolution/src/predict.py . output_reproduced --threads 8
```

From the development repository root, the equivalent command is `python3 student_resource/code/business_entity_resolution/src/predict.py student_resource student_resource/output_advanced --threads 8`. Use a new output directory for each run.

This requires a C++17 compiler (`clang++` or `g++`) and Python 3, with no third-party Python dependencies. `models/config.json` selects the correct feature schema and candidate generator. `models/pair.boost` and `models/context.boost` are portable tree ensembles.

The runner prefers `g++` when both compilers are installed; set `CXX` to choose another compiler. On hosts with a full system `/tmp`, set `TMPDIR` to a writable directory with enough space for C++ compilation. The full test run uses substantial CPU time and memory, so a server is recommended.

The runner preserves existing output directories, writes into a staging directory, checks coverage and match/candidate consistency, then publishes the completed result. `--max-queries 128` creates a smoke-test prefix only. Both indices still use full input sources; limiting the Source 1 index would change the comparison features.

The archived model in `models/` is the frozen improved model evaluated below. `output_reproduced/` is created by the command above and must match the submitted `output/` files.

## Training from the supplied data

Run these commands from the repository root. Feature caches occupy several gigabytes; development used 16 GB RAM.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r student_resource/code/business_entity_resolution/requirements.txt

ER_CODE=student_resource/code/business_entity_resolution
ER_EXP=student_resource/experiments/reproduce
mkdir -p "$ER_EXP"
clang++ -O3 -std=c++17 -o "$ER_EXP/resolver_boost" "$ER_CODE/src/boost.cpp"
clang++ -O3 -std=c++17 -o "$ER_EXP/resolver_advanced" "$ER_CODE/src/advanced.cpp"

# Original 24k pair-fit / 6k context-fit / 6k tune / 6k audit entities.
"$ER_EXP/resolver_boost" export student_resource "$ER_EXP/basic" 18000 6000 160
.venv/bin/python "$ER_CODE/src/augment_features.py" "$ER_EXP/resolver_advanced" \
  student_resource "$ER_EXP/basic" "$ER_EXP/advanced"
.venv/bin/python "$ER_CODE/src/train_boost.py" "$ER_EXP/advanced" "$ER_EXP/pair_small" \
  --iterations 1800 --depth 8 --threads 6 --calibration-queries 6000

# Expand fitting data while preserving the original tune/audit entities.
.venv/bin/python "$ER_CODE/src/expand_training.py" "$ER_EXP/resolver_advanced" \
  student_resource "$ER_EXP/advanced" "$ER_EXP/large" \
  --fit-entities 80000 --context-entities 30000
.venv/bin/python "$ER_CODE/src/train_boost.py" "$ER_EXP/large" "$ER_EXP/pair_large" \
  --iterations 1200 --depth 8 --threads 6 --learning-rate 0.035 \
  --calibration-queries 30000 --init-model "$ER_EXP/pair_small/model.cbm"
.venv/bin/python "$ER_CODE/src/prepare_reference.py" "$ER_EXP/resolver_advanced" \
  student_resource "$ER_EXP/large" "$ER_EXP/pair_large" "$ER_EXP/reference"
.venv/bin/python "$ER_CODE/src/train_context.py" "$ER_EXP/large" "$ER_EXP/pair_large" \
  "$ER_EXP/context" --iterations 1800 --depth 7 --threads 6 --entity-weight 0.5 \
  --reference-features "$ER_EXP/reference"
```

The exact selected settings are recorded in `models/validation.json`; compare them with this recipe when reproducing a particular model. Smaller configurations use the original `advanced` export and 6,000 reserved context entities.

Select the decision rule using tuning data, verify portable inference, then freeze the configuration and evaluate the audit split once:

```sh
.venv/bin/python "$ER_CODE/src/decision_policy.py" "$ER_EXP/large" "$ER_EXP/context" \
  --kind expected_f0.5
.venv/bin/python "$ER_CODE/src/verify_advanced.py" "$ER_EXP/resolver_advanced" \
  "$ER_EXP/large" "$ER_EXP/pair_large" "$ER_EXP/context" --reference-features "$ER_EXP/reference"
.venv/bin/python "$ER_CODE/src/audit_model.py" "$ER_EXP/large" "$ER_EXP/pair_large" "$ER_EXP/context"
.venv/bin/python "$ER_CODE/src/package_model.py" "$ER_EXP/large" "$ER_EXP/pair_large" \
  "$ER_EXP/context" "$ER_CODE/models"
```

Do not retune after inspecting audit results and still call that split untouched. `inspect_errors.py` deliberately refuses the audit split.

## Model details

- **Candidates:** country-aware exact/normalized name and address tokens, token pairs, normalized whole fields, consonants, and number/address blocks. A bounded retrieval pass and similarity reranking retain up to 160 candidates. Country remains an open label; no country is filtered out.
- **Pair scorer:** 107 float features describing character/token agreement, fuzzy overlap, legal suffixes, missing addresses, alphanumeric components, and approximate transliteration/sound similarities. Character rules come from Python's bundled Unicode character names and are checked into `transliteration_data.h`.
- **Context scorer:** pair logits, other candidates' confidence and source agreement, competing Source 1 scores, name frequency/rarity, and numerical error patterns. Features never contain ground-truth match counts or labels. Optional entity weighting uses training labels only as loss weights.
- **Competing references:** an index of unlabelled Source 1 records is constructed separately for train or test. Validation input fields can appear in this index, just as all test input fields do during deployment; validation labels do not participate in these features.
- **Selection:** compare the empty set with each prefix of candidates sorted by context probability. For a prefix of size `k`, approximate expected F₀.₅ with `1.25 × sum(p[:k]) / (k + 0.25 × sum(p))`; the empty-set utility is `product(1-p)`. This is an approximation, not the exact expectation of the ratio. The global threshold remains available for baseline comparisons. Evaluation includes unretrieved true links in the denominator; correctly empty singleton predictions score 1.

Training and context-fitting entities are disjoint. The larger export keeps the original 6,000 tuning and 6,000 audit entities and adds fresh fitting entities, excluding all original sampled IDs. Fixed seeds, model checksums, and exported schemas document reproducibility.

## Checks

```sh
.venv/bin/python -m unittest discover -s "$ER_CODE/src" -p 'test_*.py'
clang++ -O3 -std=c++17 -o "$ER_EXP/test_features" "$ER_CODE/src/test_features.cpp"
"$ER_EXP/test_features"
```

These cover the challenge metric, tied thresholds, unretrieved matches, entity decision sets, context isolation, cross-script names, accents, alphanumeric numbers, and digit omissions versus substitutions. The integration verifier compares both model scores and final decisions between Python and C++.

The original baseline remains reproducible using `src/main.cpp`, `model.txt`, and its `train`/`predict` commands. All approaches use only the supplied challenge data; no external business lookup, geocoding, or pretrained weights are used.
