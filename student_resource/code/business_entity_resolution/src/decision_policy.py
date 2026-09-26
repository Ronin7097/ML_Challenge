"""Choose a per-entity prediction set using approximate expected macro F0.5."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from train_boost import entity_scores, metrics


def select_matches(scores, query, threshold, policy="threshold"):
    if policy == "threshold":
        return scores >= threshold
    if policy != "expected_f0.5":
        raise ValueError(f"Unknown decision policy: {policy}")
    prediction = np.zeros(len(scores), dtype=bool)
    order = np.lexsort((-scores, query))
    sorted_query = query[order]
    boundaries = np.r_[0, np.flatnonzero(sorted_query[1:] != sorted_query[:-1]) + 1, len(order)]
    for begin, end in zip(boundaries[:-1], boundaries[1:]):
        if begin == end:
            continue
        indices = order[begin:end]
        p = 1 / (1 + np.exp(-np.clip(scores[indices], -50, 50)))
        empty = np.exp(np.log1p(-np.minimum(p, 1 - 1e-15)).sum())
        utility = 1.25 * p.cumsum() / (np.arange(1, len(p) + 1) + .25 * p.sum())
        count = int(np.argmax(np.r_[empty, utility]))
        prediction[indices[:count]] = True
    return prediction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("--kind", choices=["threshold", "expected_f0.5"], default="expected_f0.5")
    args = parser.parse_args()
    path = args.model / "metrics.json"
    report = json.loads(path.read_text())
    if "audit" in report["splits"]:
        raise ValueError("Do not change decisions after inspecting audit performance")
    schema = json.loads((args.export / "schema.json").read_text())
    with (args.export / "queries.tsv").open() as f:
        queries = list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))[schema["train_end"]:schema["tune_end"]]
    truth = np.array([int(row["truth_count"]) for row in queries])
    meta = np.memmap(args.export / "pairs.u32", dtype="<u4", mode="r").reshape(-1, 5)
    a, b = np.searchsorted(meta[:, 0], [schema["train_end"], schema["tune_end"]])
    m = meta[a:b]
    q, y = m[:, 0] - schema["train_end"], m[:, 3]
    scores = np.load(args.model / "tune_scores.npy")
    prediction = select_matches(scores, q, report["threshold_raw"], args.kind)
    result = metrics(prediction, y, q, truth)
    tp = np.bincount(q[prediction & (y == 1)], minlength=len(truth))
    fp = np.bincount(q[prediction & (y == 0)], minlength=len(truth))
    values = entity_scores(tp, fp, truth)
    baseline = entity_scores(np.array([int(r["baseline_tp"]) for r in queries]), np.array([int(r["baseline_fp"]) for r in queries]), truth)
    difference = values - baseline
    margin = 1.96 * difference.std(ddof=1) / np.sqrt(len(truth))
    result["baseline_macro_f0.5"] = float(baseline.mean())
    result["paired_improvement_95ci"] = [float(difference.mean() - margin), float(difference.mean() + margin)]
    retrieved = np.bincount(q[y == 1], minlength=len(truth))
    result["candidate_recall"] = float(retrieved.sum() / truth.sum())
    result["retrieval_oracle_macro_f0.5"] = float(entity_scores(retrieved, np.zeros(len(truth)), truth).mean())
    result["by_country"] = {country: float(values[np.array([r["country"] == country for r in queries])].mean()) for country in sorted({r["country"] for r in queries})}
    report.setdefault("threshold_tune_metrics", report["splits"]["tune"])
    report["decision_policy"] = args.kind
    report["splits"]["tune"] = result
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
