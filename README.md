# DARPA — ML Challenge 2026 submission

Business entity resolution using a fine-tuned multilingual Granite encoder and
LightGBM pair/context models. The team's reported Portal score for these
predictions is **0.970**. Local validation scores are separate measurements.

## Submit these files

- **Live Portal:** `output/matching_results.tsv`.
- **Final submission:** `DARPA_submission.zip`.

The final ZIP contains both required TSVs, runnable source, the exact trained
models, pinned dependencies, licenses, reproduction instructions and the filled
`Documentation_template.md`. Its top-level layout is:

```text
Documentation_template.md
output/
  matching_results.tsv
  candidate_pairs.tsv
code/business_entity_resolution/
  src/
  models/
  splits/
  licenses/
  validation/
  README.md
  requirements.txt
  requirements-rich.txt
  LICENSE
  run.json
  validation.json
```

Both TSVs cover **1,732,544 Source 1 entities**, including France. The official
validator passed on both files with ID checks. There are **5,958,911 predicted
links** and **99,695,890 actual scored candidate pairs** (57.543 per Source 1 on
average). Candidate size is part of final ranking; candidates have not been
trimmed after matching.

## Reproduce and verify

See [the pipeline guide](code/business_entity_resolution/README.md) for exact
commands and [the methodology](Documentation_template.md) for the model,
blocking strategy, evaluation and limitations.

From this repository root, verify the prepared package inputs or build the ZIP:

```sh
python3 code/business_entity_resolution/src/package_submission.py --root . --check-only
python3 code/business_entity_resolution/src/package_submission.py --root . --archive DARPA_submission.zip
```

The packager refuses to overwrite an existing archive. It checks output, model
and producing-code identities, then verifies the ZIP's CRC and every member's
SHA-256. The resulting `submission_manifest.json` records the final archive hash.

## GitHub and local artifacts

GitHub contains the final source, documentation, licenses, model checksums and
validation evidence. The official dataset, trained weight files, large output
TSVs, environments and ZIP are excluded from Git. The **local final ZIP includes
all trained weights and both TSVs** and is the self-contained competition
submission. For a fresh clone, copy `models/` from that ZIP into the pipeline
folder and place the official dataset at `dataset/` before running inference.

Only supplied competition records and labels were used for fitting. There are
no business lookup, geocoding or external entity-resolution API calls. The
encoder is Apache-2.0; DARPA source and original tree models are MIT.
