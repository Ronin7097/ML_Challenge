# ML Challenge 2026 — Business Entity Resolution

This repository contains a self-contained C++17 solution for the Amazon ML Challenge 2026 business entity-resolution task. Given noisy business records from three sources, it identifies the Source 2 and Source 3 entities that refer to each Source 1 entity.

The solution uses only the supplied challenge data. It does not call external data sources, geocoding services, or pretrained models.

## Repository layout

```text
student_resource/
├── code/business_entity_resolution/
│   ├── src/main.cpp                 # Training, diagnostics, and inference program
│   ├── src/validate_outputs.py      # Streaming output validator
│   ├── model.txt                    # Packaged logistic-regression parameters
│   └── README.md                    # Detailed implementation notes
├── dataset/                         # Challenge data — intentionally not tracked
├── output/                          # Generated submission TSVs — intentionally not tracked
├── utils/validate_submission.py     # Official-format validator
└── Documentation_template.md        # Submission methodology template
```

The challenge datasets, generated outputs, and submission ZIP are excluded from Git because they exceed GitHub's normal file-size limit. Obtain them from the official challenge materials and place them under `student_resource/dataset/`.

## Requirements

- A C++17 compiler (Apple Clang 14+ or GCC 9+ recommended)
- Python 3 for validation only; no third-party Python packages are required

## Quick start

From the repository root:

```sh
cd student_resource/code/business_entity_resolution
clang++ -O3 -std=c++17 -o resolver src/main.cpp

# Optional: train fresh parameters from the supplied training split.
./resolver train ../../ model.txt

# Generate the two required test-set submission files.
mkdir -p ../../output
./resolver predict ../../ model.txt ../../output
```

The prediction step writes:

- `student_resource/output/matching_results.tsv` — the leaderboard submission
- `student_resource/output/candidate_pairs.tsv` — the exact candidates scored by the matcher

Validate both outputs before packaging or uploading:

```sh
cd student_resource
python3 code/business_entity_resolution/src/validate_outputs.py \
  --source1 dataset/test/test_source1.tsv \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv
```

For the official validator, run:

```sh
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

## Method summary

1. Build an in-memory inverted index over Source 2 and Source 3 name/address tokens.
2. Retrieve up to 80 country-aware candidate records per Source 1 entity.
3. Score each candidate with name, address, token, number, legal-suffix, and script-aware features using logistic regression.
4. Apply a threshold tuned for macro F<sub>0.5</sub>, favouring precision and correctly preserving singletons.

Run `./resolver diagnose ../../` to estimate retrieval recall at several candidate-set sizes on a sampled portion of the training data.

## Reproducibility notes

- Training uses fixed random seeds.
- The packaged `model.txt` is the model used for prediction unless you retrain it.
- The `resolver` executable is a build artifact; rebuild it locally for your operating system and compiler.
- See [the implementation README](student_resource/code/business_entity_resolution/README.md) for feature and modeling detail, and fill in `student_resource/Documentation_template.md` for the final competition package.

## License

The implementation is provided under the license in `student_resource/code/business_entity_resolution/LICENSE`.
