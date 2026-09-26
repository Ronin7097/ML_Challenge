# ML Challenge 2026: Business Entity Resolution

**Team Name:** ML_Challenge  
**Team Members:** To be supplied by the submitting team  
**Submission Date:** 26 September 2026

## 1. Executive Summary

We match each Source 1 business to Source 2 and Source 3 with rare-token blocking followed by a supervised pair scorer. Blocking combines name and address words, whole normalized names, and word pairs; a logistic model makes precision-focused decisions. The system uses only the supplied data and includes every test Source 1 entity in both output files.

## 2. Methodology and Data

Training has 2,206,821 Source 1 records and 7,638,365 labeled links. Some Source 2 and Source 3 records lack addresses. Names and addresses contain abbreviations, transliterations, Indic scripts, domain names, reordered words, and deliberate near misses. Test additionally includes France, which is absent from training. We treat country as an open label used to limit comparisons to the same country; the similarity features themselves are language independent.

The pipeline loads Source 2 and Source 3, builds an in-memory sorted inverted index, then processes Source 1 one row at a time. All candidate IDs in `candidate_pairs.tsv` are the exact records passed to the matching scorer. The final matches are selected from those candidates.

## 3. Candidate Generation

For each target record, the index stores up to four name words, six address words, a compact normalized full name, and selected pairs of name and address words. At query time, rare postings are searched first within a bounded posting budget. Each retrieved target accumulates a weighted evidence score; the top 80 targets are passed to the final scorer. This allows recovery when an individual name word is common, and address keys help with aliases or names written in different scripts.

On an 18,000-entity random sample of training Source 1, this stage retrieved **57,020 of 62,300 known links (91.5%)** in the top 80. Inference produced **138,056,331 test candidate pairs**. Output rows with no candidates retain an empty ID list.

## 4. Matching Model

The final model is logistic regression trained on labeled generated candidates from 12,000 sampled training Source 1 entities. Features are name and address character-trigram Dice similarity, token overlap and Jaccard similarity, name containment, similarity interactions, numerical-address overlap and first-number agreement or conflict, exact normalized name agreement, legal-suffix-insensitive name agreement, missing-address status, and Indic-script mismatch with address similarity. The code and trained model are MIT licensed, and the model has 20 scalar coefficients.

Another 6,000 sampled Source 1 entities are held out for threshold selection and evaluation. The threshold is chosen to maximize the specified **macro F₀.₅** across all held-out entities, including singletons. The selected threshold is **0.83**. No external business data or pretrained model is used.

## 5. Results and Error Analysis

**Held-out macro F₀.₅: 0.8287.** This is a local validation estimate, not a leaderboard score. Full test inference selected **5,270,344 links**. Remaining false positives are often distinct businesses with almost identical names and addresses but a changed street or unit number. Remaining false negatives include aliases, heavy misspellings, and links with empty addresses or transliterated names. The 91.5% candidate recall limits the model's maximum possible link recall.

## 6. Reproduction

`code/business_entity_resolution/src/main.cpp` implements training and inference. The same directory contains the trained `model.txt`, `README.md` with exact build and run commands, `requirements.txt`, and an MIT license. `output/matching_results.tsv` is the leaderboard file; `output/candidate_pairs.tsv` documents the scored candidates. The standard-library C++17 implementation is deterministic for the provided inputs and fixed seeds.
