"""Package a frozen, audited model and its portable-inference verification."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("export", type=Path)
    p.add_argument("pair_model", type=Path)
    p.add_argument("context_model", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    schema = json.loads((args.export / "schema.json").read_text())
    pair = json.loads((args.pair_model / "metrics.json").read_text())
    context = json.loads((args.context_model / "metrics.json").read_text())
    parity = json.loads((args.context_model / "parity.json").read_text())
    if Path(context["configuration"]["pair_model"]).resolve() != args.pair_model.resolve():
        raise ValueError("The context model was fitted using a different pair model")
    if any(Path(report["configuration"]["export"]).resolve() != args.export.resolve() for report in [pair, context]):
        raise ValueError("The models and validation export must describe the same candidate pipeline")
    if context["features"][:schema["feature_count"]] != schema["features"]:
        raise ValueError("Pair and context feature schemas differ")
    if "audit" not in context["splits"] or not parity["identical_decisions"]:
        raise ValueError("An untouched audit evaluation and successful inference parity check are required")
    if parity.get("decision_policy", "threshold") != context.get("decision_policy", "threshold"):
        raise ValueError("Verify the final decision policy before packaging")
    args.output.mkdir(parents=True, exist_ok=True)
    for source, destination in [(args.pair_model / "model.boost", "pair.boost"), (args.context_model / "model.boost", "context.boost")]:
        shutil.copyfile(source, args.output / destination)
    digests = {name: hashlib.sha256((args.output / name).read_bytes()).hexdigest() for name in ["pair.boost", "context.boost"]}
    reserve = pair["configuration"]["calibration_queries"]
    config = {"format": "ERBOOST1", "candidate_limit": schema["candidate_limit"], "decision_policy": context.get("decision_policy", "threshold"),
              "phonetic_blocking": schema.get("phonetic_blocking", False),
              "reference_features": bool(context["configuration"].get("reference_features")),
              "pair_feature_count": schema["feature_count"], "context_feature_count": context["feature_count"],
              "pair_fit_entities": schema["train_end"] - reserve, "context_fit_entities": reserve,
              "tune_entities": schema["tune_end"] - schema["train_end"],
              "audit_entities": schema["query_count"] - schema["tune_end"], "sha256": digests}
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (args.output / "validation.json").write_text(json.dumps({"metric": "macro F0.5 including singletons and unretrieved true links",
                                                            "requested_target": .98, "target_reached_on_audit": context["splits"]["audit"]["macro_f0.5"] >= .98,
                                                            "splits": context["splits"], "parity": parity,
                                                            "pair_training": pair["configuration"], "context_training": context["configuration"]}, indent=2) + "\n")
    shutil.copyfile(args.context_model / "metrics.json", args.output / "context_metrics.json")
    shutil.copyfile(args.pair_model / "metrics.json", args.output / "pair_metrics.json")
    if (args.context_model / "audit_comparison.json").exists():
        shutil.copyfile(args.context_model / "audit_comparison.json", args.output / "audit_comparison.json")
    audit = context["splits"]["audit"]
    print(f"Packaged audited macro F0.5 {audit['macro_f0.5']:.6f} in {args.output}")


if __name__ == "__main__":
    main()
