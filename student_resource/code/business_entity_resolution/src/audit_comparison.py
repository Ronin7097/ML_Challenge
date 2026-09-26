"""Compare the frozen winner and the archived starting boost on identical audit IDs.

Run only after the winner's audit evaluation. This never tunes either model.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from catboost import CatBoostClassifier
import numpy as np

from train_boost import entity_scores, metrics
from decision_policy import select_matches


def load_audit(export):
    schema = json.loads((export / "schema.json").read_text())
    with (export / "queries.tsv").open() as f:
        queries = list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))[schema["tune_end"]:]
    meta = np.memmap(export / "pairs.u32", mode="r", dtype="<u4").reshape(-1, 5)
    first = np.searchsorted(meta[:, 0], schema["tune_end"])
    return schema, queries, meta[first:], first


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("previous_export", type=Path)
    parser.add_argument("previous_model", type=Path)
    parser.add_argument("selected_export", type=Path)
    parser.add_argument("selected_model", type=Path)
    args = parser.parse_args()
    selected_report = json.loads((args.selected_model / "metrics.json").read_text())
    if "audit" not in selected_report["splits"]:
        raise ValueError("Freeze and audit the selected model before computing this comparison")
    old_schema, old_queries, old_meta, old_first = load_audit(args.previous_export)
    schema, queries, meta, first = load_audit(args.selected_export)
    if [q["source1_entity_id"] for q in old_queries] != [q["source1_entity_id"] for q in queries]:
        raise ValueError("A paired comparison requires identical audit entities in identical order")
    truth = np.array([int(q["truth_count"]) for q in queries])
    np.testing.assert_array_equal(truth, [int(q["truth_count"]) for q in old_queries])
    old_x = np.memmap(args.previous_export / "features.f32", mode="r", dtype="<f4").reshape(-1, old_schema["feature_count"])
    old_model = CatBoostClassifier()
    old_model.load_model(str(args.previous_model / "model.cbm"))
    old_scores = old_model.predict(old_x[old_first:], prediction_type="RawFormulaVal", thread_count=6)
    old_threshold = json.loads((args.previous_model / "metrics.json").read_text())["threshold_raw"]
    old_prediction = old_scores >= old_threshold
    scores = np.load(args.selected_model / "audit_scores.npy")
    prediction = select_matches(scores, meta[:, 0], selected_report["threshold_raw"], selected_report.get("decision_policy", "threshold"))
    def evaluate(m, pred, start):
        q, labels = m[:, 0] - start, m[:, 3]
        tp = np.bincount(q[pred & (labels == 1)], minlength=len(truth))
        fp = np.bincount(q[pred & (labels == 0)], minlength=len(truth))
        return metrics(pred, labels, q, truth), entity_scores(tp, fp, truth)
    previous, previous_values = evaluate(old_meta, old_prediction, old_schema["tune_end"])
    selected, selected_values = evaluate(meta, prediction, schema["tune_end"])
    np.testing.assert_allclose(selected["macro_f0.5"], selected_report["splits"]["audit"]["macro_f0.5"], atol=1e-12)
    difference = selected_values - previous_values
    margin = 1.96 * difference.std(ddof=1) / np.sqrt(len(truth))
    report = {"previous_model": str(args.previous_model), "selected_model": str(args.selected_model),
              "previous_model_sha256": hashlib.sha256((args.previous_model / "model.cbm").read_bytes()).hexdigest(),
              "same_audit_entities": True, "previous": previous, "selected": selected,
              "macro_improvement": float(difference.mean()),
              "paired_improvement_95ci": [float(difference.mean() - margin), float(difference.mean() + margin)]}
    (args.selected_model / "audit_comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
