"""Append generic cross-script and alphanumeric features to a cached export."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("base", type=Path)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        raise ValueError("Preserve the original export in a separate directory")
    args.output.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(args.executable.resolve()), "augment", str(args.base), str(args.input), str(args.output)], check=True)
    schema = json.loads((args.input / "schema.json").read_text())
    names = json.loads((args.output / "advanced_names.json").read_text())
    old = np.memmap(args.input / "features.f32", mode="r", dtype="<f4").reshape(-1, schema["feature_count"])
    advanced = np.memmap(args.output / "advanced.f32", mode="r", dtype="<f4").reshape(-1, len(names))
    if len(old) != len(advanced):
        raise ValueError("Augmentation must preserve every pair in its original order")
    combined = np.memmap(args.output / "features.f32", mode="w+", dtype="<f4", shape=(len(old), old.shape[1] + len(names)))
    for start in range(0, len(old), 32768):
        combined[start:start + 32768, :old.shape[1]] = old[start:start + 32768]
        combined[start:start + 32768, old.shape[1]:] = advanced[start:start + 32768]
    combined.flush()
    for filename in ["pairs.u32", "queries.tsv", "truth.tsv"]:
        shutil.copyfile(args.input / filename, args.output / filename)
    schema["features"] += names
    schema["feature_count"] = len(schema["features"])
    schema["base_feature_count"] = old.shape[1]
    schema["augmentation"] = "Unicode-derived name sound features and complete numeric address components"
    (args.output / "schema.json").write_text(json.dumps(schema, indent=2) + "\n")
    print(f"Saved {len(old)} pairs with {schema['feature_count']} features", flush=True)


if __name__ == "__main__":
    main()
