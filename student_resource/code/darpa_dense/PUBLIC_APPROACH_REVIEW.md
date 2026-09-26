# Public approach review — 27 September 2026

These are architecture references, not verified top-ranked submissions. None of
the inspected repositories establishes that it belongs to the Banana team in the
user's leaderboard screenshot. Reported validation numbers are not Portal scores
and are not comparable without a shared evaluation population.

| Repository and inspected revision | Evidence and limitations | DARPA action |
| --- | --- | --- |
| [AyanAhmedKhan/amazon-ml-challenge](https://github.com/AyanAhmedKhan/amazon-ml-challenge/tree/db4640ce1b12c6d9f140acf1a6b72176ba8c14e4) | Describes five sparse retrieval views, reverse owner retrieval, competition features, and cross-fitted sibling support. Reports validation 0.98703, but its project-state review says the best run used fold-restricted competition and needs reevaluation. | Compute competition against the full owner pool. Fit successive supervised stages on separate entity groups. Test address-number and sibling features. |
| [LearnerSanket24/AmazonMLChallenge](https://github.com/LearnerSanket24/AmazonMLChallenge/tree/2267a5a7a6d7b23031e5cb66ad78e93ee263809f) | Describes hybrid retrieval and a GBDT ensemble. Its performance and ablation tables are explicitly targets/expectations, not measured evidence of a leading score. | Use as a feature checklist; do not assume its ensemble weights, runtime, or claimed reranker gain transfer. |
| [Akash-bardia/amazon-ml-challenge-2026](https://github.com/Akash-bardia/amazon-ml-challenge-2026/tree/b7d10d8f1848e66eeae57e452d15bafd29db0d43) | Reports validation macro F0.5 0.976105 with engineered features and LightGBM; describes resolving competing owners for each target. This is a repository claim, not independently reproduced. | Include house-number conflicts, missing-address indicators, and target exclusivity in development comparisons. |
| [chaXin44/amazon-ml-challenge-2026](https://github.com/chaXin44/amazon-ml-challenge-2026/tree/fb0cd690fc3f3e2661fdcf8b4cc232654a2fd1ad) | Public repository located; no comparable measured result has been verified in this review. | Do not use its existence as evidence of top performance. |

The supplied challenge statement already requires support for the unseen France
country label and permits multiple targets for one Source 1 entity. Our prepared
data enumerates countries from the records; it does not filter test data to the
two training countries. Country blocking and target-owner consistency must also
be checked against our own supplied training records.

## Implementation decisions

The new Granite encoder supplies candidates and similarity measurements. A
downstream matcher must also see lexical, address, numeric and competition
signals. Encoder loss and owner-retrieval recall are not final matching metrics.
All incoming targets, including unmatched distractors, are required to measure
singleton mistakes and macro F0.5.

The frozen DARPA model already contains substantial engineered features and
CatBoost matching. Therefore adding another generic GBDT ensemble is a lower
priority than measuring retrieval misses, rival-owner mistakes, and the
training-to-test density change. Changes will be retained only after a paired
development comparison; the sealed reserve is for the selected final challenger.

No repository code, pretrained competitor checkpoint, or submission file is
copied by this experiment. The implementation is DARPA's own code and the
permitted pinned IBM base model. GitHub's repository API did not identify a
repository license for these four revisions; README license assertions alone
are not treated as a code-reuse grant.

Primary review details: [Ayan's final recommendation](https://github.com/AyanAhmedKhan/amazon-ml-challenge/blob/db4640ce1b12c6d9f140acf1a6b72176ba8c14e4/FINAL_RECOMMENDATION.md),
[validation caveats](https://github.com/AyanAhmedKhan/amazon-ml-challenge/blob/db4640ce1b12c6d9f140acf1a6b72176ba8c14e4/PROJECT_STATE.md),
and [Sanket's expected-performance table](https://github.com/LearnerSanket24/AmazonMLChallenge/blob/2267a5a7a6d7b23031e5cb66ad78e93ee263809f/README.md#-expected-performance).
