"""Check that portable C++ inference agrees with the fitted Python model."""
import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("executable", type=Path)
    p.add_argument("export", type=Path)
    p.add_argument("model_dir", type=Path)
    args = p.parse_args()
    schema = json.loads((args.export / "schema.json").read_text())
    meta = np.memmap(args.export / "pairs.u32", dtype="<u4", mode="r").reshape(-1, 5)
    x = np.memmap(args.export / "features.f32", dtype="<f4", mode="r").reshape(-1, schema["feature_count"])
    first, last = np.searchsorted(meta[:, 0], [schema["train_end"], schema["tune_end"]])
    selected = np.random.default_rng(71).choice(np.arange(first, last), size=min(10000, last - first), replace=False)
    sample = x[selected]
    features_file = args.model_dir / "parity_features.f32"
    scores_file = args.model_dir / "parity_scores.f64"
    sample.tofile(features_file)
    subprocess.run([str(args.executable.resolve()), "score", str(args.model_dir / "model.boost"), str(features_file), str(scores_file)], check=True)
    model = CatBoostClassifier()
    model.load_model(str(args.model_dir / "model.cbm"))
    expected = model.predict(sample, prediction_type="RawFormulaVal", thread_count=2)
    actual = np.fromfile(scores_file, dtype="<f8")
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
    threshold = json.loads((args.model_dir / "metrics.json").read_text())["threshold_raw"]
    np.testing.assert_array_equal(actual >= threshold, expected >= threshold)
    result = {"pairs_checked": len(sample), "max_absolute_logit_error": float(np.abs(actual - expected).max()), "identical_decisions": True}
    (args.model_dir / "parity.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
