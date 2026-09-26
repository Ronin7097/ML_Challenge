"""Fit a candidate-context model on entities excluded from pair-model fitting.

Only predictions and feature values enter the context features. Ground-truth match
counts are used for evaluation, never as model inputs. The audit split stays sealed
unless --audit is requested after the complete configuration has been frozen.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier, Pool

from train_boost import best_threshold, entity_scores, export_model, metrics


CONTEXT_NAMES = [
    "pair_logit", "pair_probability", "score_rank", "other_best_probability",
    "other_second_probability", "other_third_probability", "other_fourth_probability",
    "other_probability_sum", "other_count_25", "other_count_50", "other_count_75",
    "other_count_90", "other_count_99", "same_source_other_best",
    "same_source_other_sum", "same_source_other_count_50", "same_source_other_count_90",
    "opposite_source_best", "opposite_source_sum", "opposite_source_count_50",
    "opposite_source_count_90", "probability_share", "probability_gap",
]


def context_features(features, scores, query, source):
    """Produce the same per-query features used by portable C++ inference."""
    result = np.empty((len(features), features.shape[1] + len(CONTEXT_NAMES)), dtype=np.float32)
    result[:, :features.shape[1]] = features
    boundaries = np.r_[0, np.flatnonzero(query[1:] != query[:-1]) + 1, len(query)]
    for begin, end in zip(boundaries[:-1], boundaries[1:]):
        logits = np.asarray(scores[begin:end], dtype=np.float64)
        p = 1 / (1 + np.exp(-np.clip(logits, -40, 40)))
        order = np.argsort(-p, kind="stable")
        rank = np.empty(len(p), dtype=np.int32)
        rank[order] = np.arange(len(p))
        best_other = np.zeros((len(p), 4))
        for i in range(len(p)):
            remaining = order[order != i][:4]
            best_other[i, :len(remaining)] = p[remaining]
        cols = [logits, p, np.log1p(rank), *best_other.T, p.sum() - p]
        for cutoff in [.25, .50, .75, .90, .99]:
            cols.append((p >= cutoff).sum() - (p >= cutoff).astype(int))
        same_best = np.zeros(len(p))
        same_sum = np.zeros(len(p))
        same_50 = np.zeros(len(p))
        same_90 = np.zeros(len(p))
        opposite_best = np.zeros(len(p))
        opposite_sum = np.zeros(len(p))
        opposite_50 = np.zeros(len(p))
        opposite_90 = np.zeros(len(p))
        for value in [2, 3]:
            same = source[begin:end] == value
            other = ~same
            same_indices = order[same[order]]
            for i in np.flatnonzero(same):
                rest = same_indices[same_indices != i]
                same_best[i] = p[rest[0]] if len(rest) else 0
            same_sum[same] = p[same].sum() - p[same]
            same_50[same] = (p[same] >= .5).sum() - (p[same] >= .5).astype(int)
            same_90[same] = (p[same] >= .9).sum() - (p[same] >= .9).astype(int)
            opposite_best[same] = p[other].max(initial=0)
            opposite_sum[same] = p[other].sum()
            opposite_50[same] = (p[other] >= .5).sum()
            opposite_90[same] = (p[other] >= .9).sum()
        cols.extend([same_best, same_sum, same_50, same_90, opposite_best,
                     opposite_sum, opposite_50, opposite_90, p / max(p.sum(), 1e-30),
                     p - best_other[:, 0]])
        result[begin:end, features.shape[1]:] = np.column_stack(cols)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export", type=Path)
    parser.add_argument("pair_model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--load", type=Path)
    parser.add_argument("--reference-features", type=Path)
    parser.add_argument("--entity-weight", type=float, default=0., help="Weight each training entity by (1 + true links)^(-power); used only in loss weights")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    schema = json.loads((args.export / "schema.json").read_text())
    pair_report = json.loads((args.pair_model / "metrics.json").read_text())
    calibration_queries = pair_report["configuration"].get("calibration_queries", 0)
    if not calibration_queries:
        raise ValueError("Pair model must reserve calibration entities with --calibration-queries")
    train_end, tune_end = schema["train_end"], schema["tune_end"]
    calibration_begin = train_end - calibration_queries
    with (args.export / "queries.tsv").open() as handle:
        queries = list(csv.DictReader(handle, delimiter="\t", quoting=csv.QUOTE_NONE))
    truth = np.array([int(q["truth_count"]) for q in queries], dtype=np.int32)
    meta = np.memmap(args.export / "pairs.u32", dtype="<u4", mode="r").reshape(-1, 5)
    features = np.memmap(args.export / "features.f32", dtype="<f4", mode="r").reshape(-1, schema["feature_count"])
    begin, middle, end = np.searchsorted(meta[:, 0], [calibration_begin, train_end, tune_end])
    pair_model = CatBoostClassifier()
    pair_model.load_model(str(args.pair_model / "model.cbm"))
    pair_scores = pair_model.predict(features[begin:end], prediction_type="RawFormulaVal", thread_count=args.threads)
    np.save(args.output / "pair_scores.npy", pair_scores)
    x = context_features(features[begin:end], pair_scores, meta[begin:end, 0], meta[begin:end, 2])
    y = meta[begin:end, 3]
    split = middle - begin
    names = schema["features"] + CONTEXT_NAMES
    reference = None
    if args.reference_features:
        reference_schema = json.loads((args.reference_features / "schema.json").read_text())
        if reference_schema["query_begin"] != calibration_begin or Path(reference_schema["pair_model"]).resolve() != args.pair_model.resolve():
            raise ValueError("Competing-reference features must use this pair model and split")
        reference = np.memmap(args.reference_features / "reference.f32", dtype="<f4", mode="r").reshape(-1, len(reference_schema["features"]))
        if len(reference) != len(meta):
            raise ValueError("Reference feature row count mismatch")
        x = np.column_stack([x, reference[begin:end]])
        names += reference_schema["features"]
    keep = (pair_scores[:split] > -8) | (y[:split] == 1) | (meta[begin:middle, 1] % 32 == 0)
    model = CatBoostClassifier(iterations=args.iterations, depth=args.depth, learning_rate=.045,
                               loss_function="Logloss", eval_metric="Logloss", l2_leaf_reg=8,
                               random_seed=20260927, thread_count=args.threads, border_count=128,
                               od_type="Iter", od_wait=120, allow_writing_files=False)
    weights = np.power(1. + truth[meta[begin:end, 0]], -args.entity_weight)
    weights /= weights[:split].mean()
    tune_pool = Pool(x[split:], y[split:], weight=weights[split:], feature_names=names)
    if args.load:
        model.load_model(str(args.load))
    else:
        print("Context training pairs", int(keep.sum()), "queries", calibration_queries, flush=True)
        model.fit(Pool(x[:split][keep], y[:split][keep], weight=weights[:split][keep], feature_names=names), eval_set=tune_pool, verbose=100)
    model.save_model(str(args.output / "model.cbm"))
    tune_scores = model.predict(tune_pool, prediction_type="RawFormulaVal", thread_count=args.threads)
    threshold, _ = best_threshold(tune_scores, y[split:], meta[middle:end, 0] - train_end, truth[train_end:tune_end])
    report = {"configuration": {"export": str(args.export), "pair_model": str(args.pair_model),
                                "calibration_begin": calibration_begin, "calibration_end": train_end,
                                "iterations": args.iterations, "depth": args.depth,
                                "entity_weight": args.entity_weight,
                                "reference_features": str(args.reference_features) if args.reference_features else None},
              "tree_count": model.tree_count_, "feature_count": len(names), "features": names,
              "threshold_raw": threshold, "threshold_probability": float(1 / (1 + np.exp(-threshold))),
              "splits": {}}
    sections = [("tune", middle, end, train_end, tune_end, tune_scores)]
    if args.audit:
        audit_pair_scores = pair_model.predict(features[end:], prediction_type="RawFormulaVal", thread_count=args.threads)
        audit_x = context_features(features[end:], audit_pair_scores, meta[end:, 0], meta[end:, 2])
        if reference is not None:
            audit_x = np.column_stack([audit_x, reference[end:]])
        audit_scores = model.predict(audit_x, prediction_type="RawFormulaVal", thread_count=args.threads)
        sections.append(("audit", end, len(meta), tune_end, len(truth), audit_scores))
    for name, start, stop, qstart, qstop, scores in sections:
        m = meta[start:stop]
        t = truth[qstart:qstop]
        q, labels = m[:, 0] - qstart, m[:, 3]
        result = metrics(scores >= threshold, labels, q, t)
        tp = np.bincount(q[(scores >= threshold) & (labels == 1)], minlength=len(t))
        fp = np.bincount(q[(scores >= threshold) & (labels == 0)], minlength=len(t))
        baseline_tp = np.array([int(r["baseline_tp"]) for r in queries[qstart:qstop]])
        baseline_fp = np.array([int(r["baseline_fp"]) for r in queries[qstart:qstop]])
        baseline = entity_scores(baseline_tp, baseline_fp, t)
        difference = entity_scores(tp, fp, t) - baseline
        margin = 1.96 * difference.std(ddof=1) / np.sqrt(len(t))
        result["baseline_macro_f0.5"] = float(baseline.mean())
        result["paired_improvement_95ci"] = [float(difference.mean() - margin), float(difference.mean() + margin)]
        retrieved = np.bincount(q[labels == 1], minlength=len(t))
        result["candidate_recall"] = float(retrieved.sum() / max(1, t.sum()))
        result["retrieval_oracle_macro_f0.5"] = float(entity_scores(retrieved, np.zeros(len(t)), t).mean())
        result["by_country"] = {country: float(entity_scores(tp, fp, t)[np.array([r["country"] == country for r in queries[qstart:qstop]])].mean())
                                for country in sorted({r["country"] for r in queries[qstart:qstop]})}
        report["splits"][name] = result
        np.save(args.output / f"{name}_scores.npy", scores)
        print(name, json.dumps(result, indent=2), flush=True)
    export_model(model, args.output / "model.boost", threshold, len(names))
    report["feature_importance"] = sorted(zip(names, model.get_feature_importance().tolist()), key=lambda pair: -pair[1])
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
