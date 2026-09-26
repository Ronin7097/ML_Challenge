"""Regenerate candidates with cross-script blocking, reusing identical pair features."""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("base", type=Path)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--candidates", type=int, default=160)
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        raise ValueError("Preserve the original candidate export")
    subprocess.run([str(args.executable.resolve()), "refresh", str(args.base), str(args.input), str(args.output), str(args.candidates)], check=True)
    schema = json.loads((args.output / "schema.json").read_text())
    schema["candidate_limit"] = args.candidates
    schema["phonetic_blocking"] = True
    (args.output / "schema.json").write_text(json.dumps(schema, indent=2) + "\n")


if __name__ == "__main__":
    main()
