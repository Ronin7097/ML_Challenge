"""Evaluate a frozen context model once, without fitting or retuning its threshold."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from catboost import CatBoostClassifier
import numpy as np

from train_boost import entity_scores, metrics
from train_context import context_features
from decision_policy import select_matches


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("export", type=Path)
    p.add_argument("pair_model", type=Path)
    p.add_argument("context_model", type=Path)
    args = p.parse_args()
    report_path = args.context_model / "metrics.json"
    report = json.loads(report_path.read_text())
    if "audit" in report["splits"]:
        raise ValueError("This configuration has already been audited; preserve that result")
    if Path(report["configuration"]["export"]).resolve() != args.export.resolve() or Path(report["configuration"]["pair_model"]).resolve() != args.pair_model.resolve():
        raise ValueError("Use the exact export and pair model recorded when fitting context")
    schema = json.loads((args.export / "schema.json").read_text())
    with (args.export / "queries.tsv").open() as f:
        queries = list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))[schema["tune_end"]:]
    truth = np.array([int(q["truth_count"]) for q in queries])
    metadata = np.memmap(args.export / "pairs.u32", dtype="<u4", mode="r").reshape(-1, 5)
    features = np.memmap(args.export / "features.f32", dtype="<f4", mode="r").reshape(-1, schema["feature_count"])
    start = np.searchsorted(metadata[:, 0], schema["tune_end"])
    meta = metadata[start:]
    pair_model = CatBoostClassifier()
    pair_model.load_model(str(args.pair_model / "model.cbm"))
    model = CatBoostClassifier()
    model.load_model(str(args.context_model / "model.cbm"))
    pair_scores = pair_model.predict(features[start:], prediction_type="RawFormulaVal", thread_count=6)
    x = context_features(features[start:], pair_scores, meta[:, 0], meta[:, 2])
    reference_dir = report["configuration"].get("reference_features")
    if reference_dir:
        reference_dir = Path(reference_dir)
        reference_schema = json.loads((reference_dir / "schema.json").read_text())
        if Path(reference_schema["pair_model"]).resolve() != args.pair_model.resolve():
            raise ValueError("Reference features belong to another pair model")
        names = reference_schema["features"]
        if report["features"][-len(names):] != names:
            raise ValueError("Reference feature schema mismatch")
        reference = np.memmap(reference_dir / "reference.f32", dtype="<f4", mode="r").reshape(-1, len(names))
        x = np.column_stack([x, reference[start:]])
    if x.shape[1] != report["feature_count"]:
        raise ValueError("Context feature schema mismatch")
    scores = model.predict(x, prediction_type="RawFormulaVal", thread_count=6)
    prediction = select_matches(scores, meta[:, 0], report["threshold_raw"], report.get("decision_policy", "threshold"))
    q, labels = meta[:, 0] - schema["tune_end"], meta[:, 3]
    result = metrics(prediction, labels, q, truth)
    tp = np.bincount(q[prediction & (labels == 1)], minlength=len(truth))
    fp = np.bincount(q[prediction & (labels == 0)], minlength=len(truth))
    values = entity_scores(tp, fp, truth)
    baseline = entity_scores(np.array([int(r["baseline_tp"]) for r in queries]),
                             np.array([int(r["baseline_fp"]) for r in queries]), truth)
    difference = values - baseline
    margin = 1.96 * difference.std(ddof=1) / np.sqrt(len(truth))
    result["baseline_macro_f0.5"] = float(baseline.mean())
    result["paired_improvement_95ci"] = [float(difference.mean() - margin), float(difference.mean() + margin)]
    retrieved = np.bincount(q[labels == 1], minlength=len(truth))
    result["candidate_recall"] = float(retrieved.sum() / truth.sum())
    result["retrieval_oracle_macro_f0.5"] = float(entity_scores(retrieved, np.zeros(len(truth)), truth).mean())
    result["by_country"] = {country: float(values[np.array([r["country"] == country for r in queries])].mean())
                            for country in sorted({r["country"] for r in queries})}
    report["splits"]["audit"] = result
    report["audit_freeze"] = {"threshold_raw": report["threshold_raw"], "decision_policy": report.get("decision_policy", "threshold"), "refitted": False,
                              "pair_model_sha256": hashlib.sha256((args.pair_model / "model.cbm").read_bytes()).hexdigest(),
                              "context_model_sha256": hashlib.sha256((args.context_model / "model.cbm").read_bytes()).hexdigest()}
    np.save(args.context_model / "audit_scores.npy", scores)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
