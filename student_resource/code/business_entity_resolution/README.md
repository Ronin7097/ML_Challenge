# Business entity resolution pipeline

This C++17 program uses only the supplied challenge data. It builds an in-memory inverted index over Source 2 and Source 3, retrieves candidates for each Source 1 record, scores those candidates with a trained logistic model, and writes both required TSV files. The model and code are MIT licensed.

## Build and reproduce

From this directory, with a C++17 compiler (tested with Apple Clang 14):

```sh
clang++ -O3 -std=c++17 -o resolver src/main.cpp
./resolver train /path/to/student_resource model.txt
mkdir -p /path/to/student_resource/output
./resolver predict /path/to/student_resource model.txt /path/to/student_resource/output
```

The base directory must contain `dataset/train/` and `dataset/test/` from the supplied files. Training uses a fixed random seed and produces `model.txt`; the packaged model file is the one used for the packaged predictions. `predict` creates `output/matching_results.tsv` and `output/candidate_pairs.tsv` for every test Source 1 row. The candidate file lists the exact top 80 candidates sent to the final scorer for each entity.

To validate both full-size output files with bounded memory:

```sh
cd /path/to/student_resource
python3 code/business_entity_resolution/src/validate_outputs.py \
  --source1 dataset/test/test_source1.tsv \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv
```

The supplied `utils/validate_submission.py` can also check the files, but it loads all candidate ID lists into memory and can require substantial memory on the full test set. The streaming checker verifies headers, row coverage and order, ID-list syntax, duplicates within lists, and that matches are a subset of candidates. The prediction code emits both files from the same target records, which guarantees target-ID existence.

## Implementation

The index uses exact name and address tokens, whole normalized names, and selected token pairs. Rare postings are searched first within a bounded per-record retrieval budget. Candidates are ordered by weighted token evidence, and the top 80 are scored. Matching features include character trigram Dice similarity, token overlap and Jaccard scores, numerical address agreement, name containment, legal-suffix-insensitive exact name agreement, missing-address status, and Indic-script mismatch. Logistic regression is trained on generated candidate pairs from 12,000 sampled training entities. A separate 6,000-entity sample tunes the decision threshold and targeted rules against macro F₀.₅. All Source 1 entities, including singletons and unseen country labels, receive one output row.

No external data lookup, geocoding, web services, or pretrained models are used.
