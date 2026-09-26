#!/usr/bin/env python3
"""Stream-check the two full-size output TSVs without building ID-list maps."""
import argparse
import itertools
import sys


def parse_list(value, label, row):
    if not value:
        return set()
    ids = value.split(",")
    if len(ids) != len(set(ids)):
        raise ValueError(f"row {row}: duplicate {label} ID")
    if any(not x.startswith(("S2-", "S3-")) for x in ids):
        raise ValueError(f"row {row}: invalid {label} ID prefix")
    return set(ids)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source1", required=True)
    p.add_argument("--matching", required=True)
    p.add_argument("--candidate", required=True)
    args = p.parse_args()
    with open(args.source1, encoding="utf-8") as src, \
         open(args.matching, encoding="utf-8") as mat, \
         open(args.candidate, encoding="utf-8") as can:
        if src.readline().rstrip("\n") != "entity_id\tbusiness_name\tbusiness_address\tcountry":
            raise ValueError("unexpected Source 1 header")
        if mat.readline().rstrip("\n") != "source1_entity_id\tmatched_entity_ids":
            raise ValueError("unexpected matching header")
        if can.readline().rstrip("\n") != "source1_entity_id\tcandidate_entity_ids":
            raise ValueError("unexpected candidate header")
        n = matches = candidates = 0
        for n, (source, match, candidate) in enumerate(itertools.zip_longest(src, mat, can), 1):
            if source is None or match is None or candidate is None:
                raise ValueError(f"row counts differ at data row {n}")
            sid = source.split("\t", 1)[0]
            ma = match.rstrip("\n").split("\t")
            ca = candidate.rstrip("\n").split("\t")
            if len(ma) != 2 or len(ca) != 2 or ma[0] != sid or ca[0] != sid:
                raise ValueError(f"malformed or misordered output at data row {n}")
            m = parse_list(ma[1], "match", n)
            c = parse_list(ca[1], "candidate", n)
            if not m.issubset(c):
                raise ValueError(f"row {n}: match absent from candidate list")
            matches += len(m)
            candidates += len(c)
    print(f"PASS: {n:,} Source 1 rows; {matches:,} matches; {candidates:,} candidates")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
