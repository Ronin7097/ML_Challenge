"""Fit on entity-disjoint splits and select decisions by the official macro F0.5.

The audit split is excluded from fitting, early stopping, and threshold selection.
Use --audit once the configuration is frozen to report that final holdout.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier, Pool


def entity_scores(tp, fp, truth):
    denom = 1.25 * tp + fp + 0.25 * (truth - tp)
    return np.divide(1.25 * tp, denom, out=(truth == 0).astype(float), where=denom != 0)


def metrics(pred, labels, query, truth):
    n = len(truth)
    tp = np.bincount(query[pred & (labels == 1)], minlength=n)
    fp = np.bincount(query[pred & (labels == 0)], minlength=n)
    values = entity_scores(tp, fp, truth)
    tp_sum, fp_sum, truth_sum = int(tp.sum()), int(fp.sum()), int(truth.sum())
    return {
        "macro_f0.5": float(values.mean()),
        "standard_error": float(values.std(ddof=1) / np.sqrt(n)),
        "precision": tp_sum / max(1, tp_sum + fp_sum),
        "recall": tp_sum / max(1, truth_sum),
        "tp": tp_sum, "fp": fp_sum, "fn": truth_sum - tp_sum,
        "entities": n,
        "singleton_accuracy": float((fp[truth == 0] == 0).mean()) if np.any(truth == 0) else None,
    }


def best_threshold(scores, labels, query, truth):
    """Exact sweep over all score ties, updating the macro objective per entity."""
    order = np.argsort(-scores, kind="stable")
    s, y, q = scores[order], labels[order], query[order]
    tp = np.zeros(len(truth), dtype=np.int64)
    fp = np.zeros(len(truth), dtype=np.int64)
    current = float(np.count_nonzero(truth == 0))
    best, threshold = current / len(truth), float(np.nextafter(s[0], np.inf))
    # A loop is cheap at validation size and handles multiple matches per query.
    for i in range(len(s)):
        j = q[i]
        old = 1.0 if truth[j] == 0 and tp[j] + fp[j] == 0 else (1.25 * tp[j] / (tp[j] + fp[j] + 0.25 * truth[j]))
        if y[i]:
            tp[j] += 1
        else:
            fp[j] += 1
        new = 1.25 * tp[j] / (tp[j] + fp[j] + 0.25 * truth[j])
        current += new - old
        if (i + 1 == len(s) or s[i] != s[i + 1]) and current / len(truth) > best:
            best, threshold = current / len(truth), float(s[i])
    return threshold, best


def export_model(model, path, threshold, feature_count):
    json_path = path.with_suffix(".json")
    model.save_model(str(json_path), format="json")
    content = json.loads(json_path.read_text())
    scale, bias = content["scale_and_bias"]
    with path.open("w") as out:
        out.write(f"ERBOOST1 {feature_count} {threshold:.17g} {scale:.17g} {bias[0]:.17g} {len(content['oblivious_trees'])}\n")
        for tree in content["oblivious_trees"]:
            out.write(f"{len(tree['splits'])}\n")
            for split in tree["splits"]:
                if split["split_type"] != "FloatFeature":
                    raise ValueError("Only float splits are supported by the C++ scorer")
                out.write(f"{split['float_feature_index']} {split['border']:.17g}\n")
            out.write(" ".join(format(x, ".17g") for x in tree["leaf_values"]) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--iterations", type=int, default=1800)
    parser.add_argument("--depth", type=int, default=7)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.06)
    parser.add_argument("--negative-rank", type=int, default=40, help="Keep all positives, top ranked negatives, and 1/16 of remaining negatives")
    parser.add_argument("--calibration-queries", type=int, default=0, help="Reserve the end of the training split for a separately fitted context model")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--load", type=Path, help="Evaluate an already fitted model")
    parser.add_argument("--init-model", type=Path, help="Continue boosting a compatible model on the current training candidates")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    schema = json.loads((args.export / "schema.json").read_text())
    with (args.export / "queries.tsv").open() as handle:
        queries = list(csv.DictReader(handle, delimiter="\t", quoting=csv.QUOTE_NONE))
    truth = np.array([int(q["truth_count"]) for q in queries], dtype=np.int32)
    meta = np.memmap(args.export / "pairs.u32", dtype="<u4", mode="r").reshape(-1, 5)
    x = np.memmap(args.export / "features.f32", dtype="<f4", mode="r").reshape(-1, schema["feature_count"])
    if len(meta) != len(x) or len(schema["features"]) != x.shape[1]:
        raise ValueError("Feature/metadata schema mismatch")
    train_end, tune_end = schema["train_end"], schema["tune_end"]
    train_rows = int(np.searchsorted(meta[:, 0], train_end))
    fit_rows = int(np.searchsorted(meta[:, 0], train_end - args.calibration_queries))
    tune_rows = int(np.searchsorted(meta[:, 0], tune_end))
    rank_column = schema["features"].index("retrieval_rank")
    selected = (meta[:fit_rows, 3] == 1) | (x[:fit_rows, rank_column] <= np.log1p(args.negative_rank)) | (meta[:fit_rows, 1] % 16 == 0)
    print("Training on", int(selected.sum()), "of", fit_rows, "pairs; all validation candidates retained", flush=True)
    train_pool = Pool(x[:fit_rows][selected], meta[:fit_rows, 3][selected], feature_names=schema["features"])
    tune_pool = Pool(x[train_rows:tune_rows], meta[train_rows:tune_rows, 3], feature_names=schema["features"])
    model = CatBoostClassifier(
        iterations=args.iterations, depth=args.depth, learning_rate=args.learning_rate,
        loss_function="Logloss", eval_metric="Logloss", random_seed=20260926,
        l2_leaf_reg=6, thread_count=args.threads, border_count=128,
        od_type="Iter", od_wait=150, allow_writing_files=False,
    )
    if args.load:
        model.load_model(str(args.load))
    else:
        model.fit(train_pool, eval_set=tune_pool, verbose=100, init_model=str(args.init_model) if args.init_model else None)
    model.save_model(str(args.output / "model.cbm"))
    tune_scores = model.predict(tune_pool, prediction_type="RawFormulaVal", thread_count=args.threads)
    threshold, _ = best_threshold(tune_scores, meta[train_rows:tune_rows, 3], meta[train_rows:tune_rows, 0] - train_end, truth[train_end:tune_end])
    report = {"configuration": {**vars(args), "export": str(args.export), "output": str(args.output), "load": str(args.load) if args.load else None,
                                "init_model": str(args.init_model) if args.init_model else None},
              "tree_count": model.tree_count_, "threshold_raw": threshold, "threshold_probability": float(1 / (1 + np.exp(-threshold))), "splits": {}}
    sections = [("tune", train_rows, tune_rows, train_end, tune_end)]
    if args.audit:
        sections.append(("audit", tune_rows, len(meta), tune_end, len(truth)))
    for name, begin, end, qbegin, qend in sections:
        if qend == qbegin:
            continue
        scores = tune_scores if name == "tune" else model.predict(x[begin:end], prediction_type="RawFormulaVal", thread_count=args.threads)
        m, t = meta[begin:end], truth[qbegin:qend]
        q, y = m[:, 0] - qbegin, m[:, 3]
        baseline_tp = np.array([int(r["baseline_tp"]) for r in queries[qbegin:qend]])
        baseline_fp = np.array([int(r["baseline_fp"]) for r in queries[qbegin:qend]])
        retrieved = np.bincount(q[y == 1], minlength=len(t))
        section = metrics(scores >= threshold, y, q, t)
        section["baseline_macro_f0.5"] = float(entity_scores(baseline_tp, baseline_fp, t).mean())
        section["candidate_recall"] = float(retrieved.sum() / t.sum())
        section["retrieval_oracle_macro_f0.5"] = float(entity_scores(retrieved, np.zeros(len(t)), t).mean())
        section["by_country"] = {}
        for country in sorted({r["country"] for r in queries[qbegin:qend]}):
            mask = np.array([r["country"] == country for r in queries[qbegin:qend]])
            tp = np.bincount(q[(scores >= threshold) & (y == 1)], minlength=len(t))
            fp = np.bincount(q[(scores >= threshold) & (y == 0)], minlength=len(t))
            section["by_country"][country] = float(entity_scores(tp, fp, t)[mask].mean())
        report["splits"][name] = section
        np.save(args.output / f"{name}_scores.npy", scores)
        print(name, json.dumps(section, indent=2), flush=True)
    export_model(model, args.output / "model.boost", threshold, x.shape[1])
    importance = sorted(zip(schema["features"], model.get_feature_importance()), key=lambda pair: -pair[1])
    report["feature_importance"] = [(name, float(value)) for name, value in importance]
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print("Saved", args.output / "model.boost", "threshold", threshold, flush=True)


if __name__ == "__main__":
    main()
