"""Join selected development errors to source records without loading all targets.

This deliberately refuses the audit split: inspecting it would turn it into tuning data.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("base", type=Path)
    p.add_argument("export", type=Path)
    p.add_argument("--model-dir", type=Path)
    p.add_argument("--kind", choices=["missed", "fp", "fn"], default="missed")
    p.add_argument("--split", choices=["train", "tune"], default="tune")
    p.add_argument("--limit", type=int, default=20)
    args = p.parse_args()
    with (args.export / "queries.tsv").open() as f:
        queries = {int(q["query"]): q for q in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE) if q["split"] == args.split}
    errors = []
    if args.kind == "missed":
        with (args.export / "truth.tsv").open() as f:
            for row in csv.DictReader(f, delimiter="\t"):
                if row["retrieved"] == "0" and int(row["query"]) in queries:
                    errors.append((int(row["query"]), row["matched_entity_id"], None))
                    if len(errors) >= args.limit:
                        break
    else:
        if args.model_dir is None or args.split != "tune":
            p.error("fp/fn inspection requires --model-dir and --split tune")
        schema = json.loads((args.export / "schema.json").read_text())
        report = json.loads((args.model_dir / "metrics.json").read_text())
        meta = np.memmap(args.export / "pairs.u32", dtype="<u4", mode="r").reshape(-1, 5)
        begin, end = np.searchsorted(meta[:, 0], [schema["train_end"], schema["tune_end"]])
        meta = meta[begin:end]
        scores = np.load(args.model_dir / "tune_scores.npy")
        pred = scores >= report["threshold_raw"]
        wrong = pred & (meta[:, 3] == 0) if args.kind == "fp" else (~pred) & (meta[:, 3] == 1)
        candidates = np.flatnonzero(wrong)
        # Show typical errors, not just the most extreme scores.
        np.random.default_rng(69).shuffle(candidates)
        for i in candidates[:args.limit]:
            q, record, source = meta[i, :3]
            errors.append((int(q), f"S{source}-{record}", float(1 / (1 + np.exp(-scores[i])))))
    wanted = {record for _, record, _ in errors}
    targets = {}
    for source in [2, 3]:
        with (args.base / f"dataset/train/train_source{source}.tsv").open() as f:
            next(f)
            for line in f:
                fields = line.rstrip("\n").split("\t")
                if fields[0] in wanted:
                    targets[fields[0]] = fields[1:]
    for q, record, score in errors:
        query = queries[q]
        print(json.dumps({"kind": args.kind, "probability": score,
                          "query": [query["source1_entity_id"], query["name"], query["address"]],
                          "target": [record, *targets.get(record, [])]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
