"""Add fresh training entities while preserving the original tuning/audit sets."""
import argparse
import csv
import json
from pathlib import Path
import random
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("executable", type=Path)
    p.add_argument("base", type=Path)
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--fit-entities", type=int, default=80000)
    p.add_argument("--context-entities", type=int, default=30000)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plan = args.output / "plan"
    plan.mkdir(exist_ok=True)
    schema = json.loads((args.input / "schema.json").read_text())
    with (args.input / "queries.tsv").open() as f:
        original = list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
    # The existing pair model used 24k entities and reserved the following 6k.
    if schema["train_end"] != 30000 or schema["tune_end"] != 36000 or len(original) != 42000:
        raise ValueError("Expected the preserved 24k/6k/6k/6k starting split")
    extra_fit, extra_context = args.fit_entities - 24000, args.context_entities - 6000
    if min(extra_fit, extra_context) < 0:
        raise ValueError("Expansion cannot remove previously reserved entities")
    excluded = {row["source1_entity_id"] for row in original}
    rng = random.Random(20260928)
    reservoir = []
    wanted = extra_fit + extra_context
    seen = 0
    with (args.base / "dataset/train/train_ground_truth.tsv").open() as f:
        next(f)
        for line in f:
            entity, ids = line.rstrip("\n").split("\t")
            if entity in excluded:
                continue
            seen += 1
            if len(reservoir) < wanted:
                reservoir.append((entity, ids))
            else:
                index = rng.randrange(seen)
                if index < wanted:
                    reservoir[index] = (entity, ids)
    if len(reservoir) != wanted:
        raise ValueError("Insufficient new entities")
    rng.shuffle(reservoir)
    def fresh(items):
        return [{"source1_entity_id": entity, "split": "train", "truth_count": str(len(ids.split(",")) if ids else 0),
                 "baseline_tp": "0", "baseline_fp": "0", "baseline_retrieved": "0", "truth_ids": ids} for entity, ids in items]
    rows = (original[:24000] + fresh(reservoir[:extra_fit]) + original[24000:30000]
            + fresh(reservoir[extra_fit:]) + original[30000:])
    by_id = {row["source1_entity_id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("Entity split overlap")
    old_truth = [[] for _ in original]
    with (args.input / "truth.tsv").open() as f:
        for item in csv.DictReader(f, delimiter="\t"):
            old_truth[int(item["query"])].append(item["matched_entity_id"])
    for old, ids in zip(original, old_truth):
        old["truth_ids"] = ",".join(ids)
    with (args.base / "dataset/train/train_source1.tsv").open() as f:
        next(f)
        for line in f:
            entity, name, address, country = line.rstrip("\n").split("\t")
            if entity in by_id:
                by_id[entity].update(name=name, address=address, country=country)
    fields = ["query", "source1_entity_id", "split", "truth_count", "country", "baseline_tp", "baseline_fp", "baseline_retrieved", "name", "address"]
    with (plan / "queries.tsv").open("w") as qfile, (plan / "truth.tsv").open("w") as tfile:
        qfile.write("\t".join(fields) + "\n")
        tfile.write("query\tmatched_entity_id\tretrieved\n")
        for index, row in enumerate(rows):
            row["query"] = str(index)
            if "country" not in row:
                raise ValueError("Missing Source 1 record")
            qfile.write("\t".join(row[field] for field in fields) + "\n")
            for entity in row["truth_ids"].split(","):
                if entity:
                    tfile.write(f"{index}\t{entity}\t0\n")
    schema.update(train_end=args.fit_entities + args.context_entities,
                  tune_end=args.fit_entities + args.context_entities + 6000,
                  query_count=len(rows), fit_entities=args.fit_entities,
                  calibration_queries=args.context_entities, phonetic_blocking=False)
    (plan / "schema.json").write_text(json.dumps(schema, indent=2) + "\n")
    print(f"Exporting {len(rows)} queries: {args.fit_entities} pair fit, {args.context_entities} context fit, 6000 tune, 6000 audit", flush=True)
    subprocess.run([str(args.executable.resolve()), "expand", str(args.base), str(args.input), str(plan), str(args.output)], check=True)


if __name__ == "__main__":
    main()
