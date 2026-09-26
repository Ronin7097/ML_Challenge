"""Export unlabelled competing-reference features for reserved context entities."""
import argparse
import json
from pathlib import Path
import subprocess

from catboost import CatBoostClassifier
import numpy as np

REFERENCE_NAMES = ["reference_checked", "reference_best_logit", "reference_second_logit",
                   "reference_logit_margin", "reference_better_count", "reference_confident_count",
                   "reference_query_name_count", "reference_target_name_count",
                   "name_idf_fuzzy_query", "name_idf_fuzzy_target", "name_idf_exact_query",
                   "name_idf_exact_target", "name_rarest_query_similarity", "name_rarest_target_similarity",
                   "first_digits_length_change", "first_digits_length_ratio", "first_digits_containment", "first_digits_subsequence",
                   "first_numeric_distance", "first_numeric_one_apart", "digit_omission_query", "digit_omission_target",
                   "unmatched_long_query_numbers", "unmatched_long_target_numbers", "unmatched_short_query_numbers", "unmatched_short_target_numbers"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("executable", type=Path)
    p.add_argument("base", type=Path)
    p.add_argument("export", type=Path)
    p.add_argument("model", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    schema = json.loads((args.export / "schema.json").read_text())
    report = json.loads((args.model / "metrics.json").read_text())
    reserve = report["configuration"].get("calibration_queries", 0)
    if not reserve:
        raise ValueError("Reserve context training entities before fitting the pair model")
    start = schema["train_end"] - reserve
    x = np.memmap(args.export / "features.f32", dtype="<f4", mode="r").reshape(-1, schema["feature_count"])
    model = CatBoostClassifier()
    model.load_model(str(args.model / "model.cbm"))
    scores_path = args.output / "pair_scores.f64"
    with scores_path.open("wb") as out:
        for begin in range(0, len(x), 65536):
            scores = model.predict(x[begin:begin + 65536], prediction_type="RawFormulaVal", thread_count=6)
            np.asarray(scores, dtype="<f8").tofile(out)
    subprocess.run([str(args.executable.resolve()), "reverse", str(args.base), str(args.export),
                    str(args.model / "model.boost"), str(scores_path), str(args.output / "reference.f32"), str(start)], check=True)
    expected = len(x) * len(REFERENCE_NAMES) * 4
    if (args.output / "reference.f32").stat().st_size != expected:
        raise ValueError("Reference export must preserve pair order and length")
    (args.output / "schema.json").write_text(json.dumps({"features": REFERENCE_NAMES, "query_begin": start,
                                                         "pair_model": str(args.model), "export": str(args.export)}, indent=2) + "\n")


if __name__ == "__main__":
    main()
