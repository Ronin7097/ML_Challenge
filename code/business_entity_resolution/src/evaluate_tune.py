"""Exact fresh-tune macro F0.5, optional paired name-group bootstrap."""
import argparse
import csv
import json
from pathlib import Path

import duckdb
import numpy as np


def read_predictions(path, wanted, value_column="matched_entity_ids"):
    found = {}
    with path.open(newline="") as f:
        rows = csv.DictReader(f, delimiter="\t")
        if rows.fieldnames != ["source1_entity_id", value_column]:
            raise ValueError("Invalid prediction header")
        for row in rows:
            q = row["source1_entity_id"]
            if q not in wanted:
                continue
            if q in found:
                raise ValueError("Duplicate query")
            ids = row[value_column].split(",") if row[value_column] else []
            if len(ids) != len(set(ids)) or any(not x.startswith(("S2-", "S3-")) for x in ids):
                raise ValueError("Malformed target IDs")
            found[q] = set(ids)
    if found.keys() != wanted:
        raise ValueError("Prediction does not cover exactly every requested tune query")
    return found


def entity_scores(gold, predictions, ids):
    scores, tp, fp, fn = [], 0, 0, 0
    for q in ids:
        truth, pred = gold[q], predictions[q]
        common = len(truth & pred)
        scores.append(1.25*common/(0.25*len(truth)+len(pred)) if truth or pred else 1.0)
        tp += common; fp += len(pred)-common; fn += len(truth)-common
    return np.asarray(scores), {"tp": tp, "fp": fp, "fn": fn}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--prediction", type=Path, required=True)
    p.add_argument("--baseline", type=Path)
    p.add_argument("--candidate", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--role", choices=["tune", "reserve"], default="tune")
    a = p.parse_args()
    db = duckdb.connect(str(a.data/"records.duckdb"), read_only=True)
    rows = db.execute("""WITH frequencies AS (SELECT name_group,count(*) AS n FROM roles GROUP BY 1)
      SELECT r.entity_id,r.country,r.name_group,f.n,t.truth_ids FROM roles r
      JOIN frequencies f USING(name_group) JOIN truth t ON r.entity_id=t.s1_id
      WHERE r.role=? ORDER BY r.entity_id""", [a.role]).fetchall()
    ids = [r[0] for r in rows]
    wanted = set(ids)
    gold = {r[0]: set(r[4].split(',')) if r[4] else set() for r in rows}
    pred = read_predictions(a.prediction, wanted)
    score, links = entity_scores(gold, pred, ids)
    singleton = np.asarray([not gold[q] for q in ids])
    report = {"evaluation": ("name-group development tune; not Portal" if a.role=="tune"
                             else "name-group reserve comparison; not Portal; consult experiment history for prior use"),
              "entities": len(ids),
              "macro_f05": float(score.mean()), "links": links,
              "singleton_entities": int(singleton.sum()),
              "singleton_accuracy": float(score[singleton].mean()) if singleton.any() else None}
    if a.candidate:
        candidates=read_predictions(a.candidate,wanted,"candidate_entity_ids")
        if any(not pred[q].issubset(candidates[q]) for q in ids):
            raise ValueError("Match outside actual candidate set")
        recovered={q:gold[q]&candidates[q] for q in ids}
        oracle,counts=entity_scores(gold,recovered,ids)
        report['retrieval']={'candidate_pairs':sum(len(candidates[q]) for q in ids),
                             'true_links_retrieved':counts['tp'],'true_links_missed':counts['fn'],
                             'recall':counts['tp']/(counts['tp']+counts['fn']),
                             'oracle_macro_f05':float(oracle.mean())}
    countries = np.asarray([r[1] for r in rows])
    frequency = np.asarray([r[3] for r in rows])
    report["strata"] = {}
    for key, mask in [(f"country:{c}", countries==c) for c in sorted(set(countries))]+[
        ("name_frequency:1", frequency==1), ("name_frequency:2-5", (frequency>=2)&(frequency<=5)),
        ("name_frequency:6+", frequency>=6)]:
        report["strata"][key] = {"entities": int(mask.sum()), "macro_f05": float(score[mask].mean()) if mask.any() else None}
    if a.baseline:
        base, _ = entity_scores(gold, read_predictions(a.baseline, wanted), ids)
        delta = score-base
        _, group = np.unique([r[2] for r in rows], return_inverse=True)
        count = np.bincount(group)
        sums = np.bincount(group, weights=delta)
        rng = np.random.default_rng(20260927)
        means = []
        for _ in range(2000):
            sample = rng.integers(len(count), size=len(count))
            means.append(float(sums[sample].sum()/count[sample].sum()))
        report["paired"] = {"baseline_macro_f05": float(base.mean()), "delta": float(delta.mean()),
                            "name_groups": len(count), "bootstrap_replicates": 2000,
                            "delta_95pct_interval": np.quantile(means, [0.025,0.975]).tolist()}
        for key in report["strata"]:
            if key.startswith("country:"): mask=countries==key.split(":",1)[1]
            elif key.endswith(":1"): mask=frequency==1
            elif key.endswith(":2-5"): mask=(frequency>=2)&(frequency<=5)
            else: mask=frequency>=6
            report["strata"][key]["baseline_macro_f05"] = float(base[mask].mean()) if mask.any() else None
            report["strata"][key]["paired_delta"] = float(delta[mask].mean()) if mask.any() else None
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
