"""Verify Python/C++ score and grouped-feature parity on development entities."""
import argparse
import json
from pathlib import Path
import subprocess

from catboost import CatBoostClassifier
import numpy as np

from train_context import context_features
from decision_policy import select_matches


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("executable", type=Path)
    p.add_argument("export", type=Path)
    p.add_argument("pair_model", type=Path)
    p.add_argument("context_model", type=Path)
    p.add_argument("--reference-features", type=Path)
    args = p.parse_args()
    schema = json.loads((args.export / "schema.json").read_text())
    meta = np.memmap(args.export / "pairs.u32", dtype="<u4", mode="r").reshape(-1, 5)
    x = np.memmap(args.export / "features.f32", dtype="<f4", mode="r").reshape(-1, schema["feature_count"])
    output = args.context_model / "verification"
    output.mkdir(exist_ok=True)
    pair_model = CatBoostClassifier()
    pair_model.load_model(str(args.pair_model / "model.cbm"))
    context_model = CatBoostClassifier()
    context_model.load_model(str(args.context_model / "model.cbm"))
    context_report = json.loads((args.context_model / "metrics.json").read_text())
    policy = context_report.get("decision_policy", "threshold")
    reference = None
    if args.reference_features:
        names = json.loads((args.reference_features / "schema.json").read_text())["features"]
        reference = np.memmap(args.reference_features / "reference.f32", dtype="<f4", mode="r").reshape(-1, len(names))
    rng = np.random.default_rng(102)
    queries = rng.choice(np.arange(schema["train_end"], schema["tune_end"]), size=20, replace=False)
    pair_error = 0
    context_error = 0
    checked = 0
    for query in queries:
        start, stop = np.searchsorted(meta[:, 0], [query, query + 1])
        features = np.array(x[start:stop])
        source = np.array(meta[start:stop, 2], dtype="u1")
        expected_pair = pair_model.predict(features, prediction_type="RawFormulaVal", thread_count=2)
        features.tofile(output / "features.f32")
        expected_pair.tofile(output / "pair.f64")
        source.tofile(output / "source.u8")
        subprocess.run([str(args.executable.resolve()), "score", str(args.pair_model / "model.boost"),
                        str(output / "features.f32"), str(output / "cpp_pair.f64"), str(features.shape[1])], check=True)
        actual_pair = np.fromfile(output / "cpp_pair.f64", dtype="<f8")
        np.testing.assert_allclose(actual_pair, expected_pair, rtol=1e-10, atol=1e-10)
        pair_error = max(pair_error, float(np.max(np.abs(actual_pair - expected_pair))))
        subprocess.run([str(args.executable.resolve()), "context", str(output / "features.f32"),
                        str(output / "cpp_pair.f64"), str(output / "source.u8"), str(output / "cpp_context.f32"), str(len(features))], check=True)
        expected_x = context_features(features, expected_pair, meta[start:stop, 0], source)
        actual_x = np.fromfile(output / "cpp_context.f32", dtype="<f4").reshape(expected_x.shape)
        np.testing.assert_allclose(actual_x, expected_x, rtol=1e-6, atol=1e-7)
        if reference is not None:
            expected_x = np.column_stack([expected_x, reference[start:stop]])
            actual_x = np.column_stack([actual_x, reference[start:stop]])
        actual_x.tofile(output / "cpp_context.f32")
        subprocess.run([str(args.executable.resolve()), "score", str(args.context_model / "model.boost"),
                        str(output / "cpp_context.f32"), str(output / "cpp_final.f64"), str(actual_x.shape[1])], check=True)
        expected_final = context_model.predict(expected_x, prediction_type="RawFormulaVal", thread_count=2)
        actual_final = np.fromfile(output / "cpp_final.f64", dtype="<f8")
        np.testing.assert_allclose(actual_final, expected_final, rtol=1e-10, atol=1e-10)
        threshold = context_report["threshold_raw"]
        subprocess.run([str(args.executable.resolve()), "decide", str(output / "cpp_final.f64"), str(output / "cpp_decisions.u8"),
                        str(len(features)), str(threshold), str(int(policy == "expected_f0.5"))], check=True)
        actual_decisions = np.fromfile(output / "cpp_decisions.u8", dtype="u1").astype(bool)
        expected_decisions = select_matches(expected_final, meta[start:stop, 0], threshold, policy)
        np.testing.assert_array_equal(actual_decisions, expected_decisions)
        context_error = max(context_error, float(np.max(np.abs(actual_final - expected_final))))
        checked += len(features)
    result = {"queries": len(queries), "pairs": checked, "pair_max_absolute_error": pair_error,
              "context_max_absolute_error": context_error, "identical_decisions": True, "decision_policy": policy}
    (args.context_model / "parity.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
