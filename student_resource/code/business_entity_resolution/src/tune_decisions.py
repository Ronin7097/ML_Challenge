"""Compare a global threshold with an entity-level expected-F0.5 decision rule.

This uses only the tuning split. It never opens audit predictions or labels.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np

from train_boost import entity_scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export", type=Path)
    parser.add_argument("model", type=Path)
    args = parser.parse_args()
    schema = json.loads((args.export / "schema.json").read_text())
    report = json.loads((args.model / "metrics.json").read_text())
    meta = np.memmap(args.export / "pairs.u32", mode="r", dtype="<u4").reshape(-1, 5)
    a, b = np.searchsorted(meta[:, 0], [schema["train_end"], schema["tune_end"]])
    m = meta[a:b]
    scores = np.load(args.model / "tune_scores.npy")
    with (args.export / "queries.tsv").open() as f:
        queries = list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))[schema["train_end"]:schema["tune_end"]]
    truth = np.array([int(q["truth_count"]) for q in queries])
    q = m[:, 0] - schema["train_end"]
    counts = np.bincount(q, minlength=len(truth))
    width = int(counts.max())
    matrix = np.full((len(truth), width), -100., dtype=np.float64)
    labels = np.zeros_like(matrix)
    offsets = np.r_[0, np.cumsum(counts)[:-1]]
    columns = np.arange(len(q)) - offsets[q]
    matrix[q, columns] = scores
    labels[q, columns] = m[:, 3]
    order = np.argsort(-matrix, axis=1, kind="stable")
    matrix = np.take_along_axis(matrix, order, axis=1)
    labels = np.take_along_axis(labels, order, axis=1)
    prefix = np.column_stack([np.zeros(len(truth)), labels.cumsum(axis=1)])
    k = np.arange(1, width + 1)[None, :]
    results = []
    for scale in [.5, .75, 1., 1.25, 1.5, 2.]:
        for shift in np.arange(-2., 2.01, .5):
            p = 1 / (1 + np.exp(-np.clip(matrix * scale + shift, -50, 50)))
            empty = np.exp(np.log1p(-np.minimum(p, 1 - 1e-15)).sum(axis=1))
            utility = 1.25 * p.cumsum(axis=1) / (k + .25 * p.sum(axis=1)[:, None])
            choice = np.argmax(np.column_stack([empty, utility]), axis=1)
            tp = prefix[np.arange(len(truth)), choice]
            score = float(entity_scores(tp, choice - tp, truth).mean())
            results.append({"scale": scale, "shift": float(shift), "macro_f0.5": score})
    results.sort(key=lambda r: -r["macro_f0.5"])
    output = {"baseline": report["splits"]["tune"]["macro_f0.5"], "best": results[0], "top_five": results[:5]}
    (args.model / "decision_experiment.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
