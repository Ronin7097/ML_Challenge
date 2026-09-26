"""Build the portable matcher and reproduce predictions from a verified model bundle."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("base", type=Path, help="student_resource directory")
    p.add_argument("output", type=Path, help="New output directory; existing outputs are preserved")
    p.add_argument("--models", type=Path, default=Path(__file__).resolve().parent.parent / "models")
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--max-queries", type=int, default=0, help="Nonzero creates only a smoke-test prefix, not a complete submission")
    args = p.parse_args()
    if args.output.exists():
        p.error("Choose a new output directory so existing predictions are preserved")
    if not 1 <= args.threads <= 16 or args.max_queries < 0:
        p.error("threads must be 1..16 and max-queries must be nonnegative")
    config = json.loads((args.models / "config.json").read_text())
    for filename, expected in config["sha256"].items():
        if hashlib.sha256((args.models / filename).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Model checksum mismatch: {filename}")
    source = Path(__file__).resolve().parent
    executable = source.parent / "resolver_advanced"
    compiler = os.environ.get("CXX") or shutil.which("clang++") or shutil.which("g++")
    if not compiler:
        raise RuntimeError("A C++17 compiler is required")
    if not executable.exists() or any(f.stat().st_mtime > executable.stat().st_mtime for f in source.iterdir() if f.suffix in {".cpp", ".h"}):
        subprocess.run([compiler, "-O3", "-std=c++17", "-o", str(executable), str(source / "advanced.cpp")], check=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{args.output.name}-", dir=args.output.parent))
    environment = dict(os.environ, ER_THREADS=str(args.threads))
    subprocess.run([str(executable), "predict", str(args.base), str(args.models / "pair.boost"),
                    str(args.models / "context.boost"), str(staging), str(int(config["reference_features"])),
                    str(config["candidate_limit"]), str(args.max_queries), str(int(config["phonetic_blocking"]))],
                   env=environment, check=True)
    source1 = args.base / "dataset/test/test_source1.tsv"
    if args.max_queries:
        source1 = staging / "source1_prefix.tsv"
        with (args.base / "dataset/test/test_source1.tsv").open() as src, source1.open("w") as dst:
            for index, line in enumerate(src):
                if index > args.max_queries:
                    break
                dst.write(line)
    subprocess.run([os.sys.executable, str(source / "validate_outputs.py"), "--source1", str(source1),
                    "--matching", str(staging / "matching_results.tsv"), "--candidate", str(staging / "candidate_pairs.tsv")], check=True)
    (staging / "run.json").write_text(json.dumps({"model_sha256": config["sha256"], "max_queries": args.max_queries,
                                                  "complete_submission": args.max_queries == 0}, indent=2) + "\n")
    if args.output.exists():
        raise FileExistsError(f"Output appeared during prediction; completed files remain in {staging}")
    staging.rename(args.output)
    print(f"Validated {'smoke-test prefix' if args.max_queries else 'complete predictions'} saved in {args.output}")


if __name__ == "__main__":
    main()
