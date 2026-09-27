"""Run the unmodified official validator on both TSVs and record blocking cost."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def quantile(histogram, count, fraction):
    rank = max(1, math.ceil(count * fraction))
    cumulative = 0
    for value, frequency in sorted(histogram.items()):
        cumulative += frequency
        if cumulative >= rank:
            return value
    raise ValueError('Empty candidate histogram')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise ValueError('Refuse to overwrite an audit report')
    started = time.monotonic()
    official = Path(__file__).with_name('validate_submission_official.py')
    matching = args.output / 'matching_results.tsv'
    candidate = args.output / 'candidate_pairs.tsv'
    for path in (official, matching, candidate):
        if not path.is_file():
            raise FileNotFoundError(path)
    command = [sys.executable, '-u', str(official), '--matching', str(matching),
               '--candidate', str(candidate), '--test-dir', str(args.test_dir), '--check-ids']
    completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(completed.stdout, flush=True)
    if completed.returncode or 'PASS' not in completed.stdout or 'WARNING' in completed.stdout:
        raise RuntimeError('Full official validation did not pass without warnings')

    histogram = Counter()
    countries = {}
    with (args.test_dir / 'test_source1.tsv').open() as source, matching.open() as mat, candidate.open() as can:
        next(source); next(mat); next(can)
        for row in source:
            sid = row.split('\t', 1)[0]
            country = row.rstrip('\n').rsplit('\t', 1)[1]
            mr, cr = mat.readline(), can.readline()
            if not mr or not cr:
                raise ValueError('Output ended early')
            mid, mlist = mr.rstrip('\n').split('\t')
            cid, clist = cr.rstrip('\n').split('\t')
            if sid != mid or sid != cid:
                raise ValueError('Source/output row order differs')
            matches = len(mlist.split(',')) if mlist else 0
            candidates = len(clist.split(',')) if clist else 0
            if not set(mlist.split(',') if mlist else ()).issubset(clist.split(',') if clist else ()):
                raise ValueError('A match is absent from its candidate list')
            histogram[candidates] += 1
            stats = countries.setdefault(country, dict(source1_rows=0, target_rows=0, matches=0, candidate_pairs=0))
            stats['source1_rows'] += 1
            stats['matches'] += matches
            stats['candidate_pairs'] += candidates
        if mat.readline() or can.readline():
            raise ValueError('Extra output rows')
    for name in ('test_source2.tsv', 'test_source3.tsv'):
        with (args.test_dir / name).open() as stream:
            next(stream)
            for row in stream:
                country = row.rstrip('\n').rsplit('\t', 1)[1]
                stats = countries.setdefault(country, dict(source1_rows=0, target_rows=0, matches=0, candidate_pairs=0))
                stats['target_rows'] += 1
    rows = sum(histogram.values())
    pairs = sum(value * count for value, count in histogram.items())
    target_rows = sum(c['target_rows'] for c in countries.values())
    country_cartesian = sum(c['source1_rows'] * c['target_rows'] for c in countries.values())
    for stats in countries.values():
        stats['mean_candidates_per_source1'] = stats['candidate_pairs'] / stats['source1_rows'] if stats['source1_rows'] else None
    report = dict(
        official_both_files_and_ids='PASS', match_subset='PASS', source1_order='PASS',
        official_command=command, official_output=completed.stdout.strip(),
        official_validator_sha256=digest(official),
        matching_sha256=digest(matching), candidate_sha256=digest(candidate),
        test_sha256={p.name: digest(p) for p in sorted(args.test_dir.glob('test_source*.tsv'))},
        source1_rows=rows, target_rows=target_rows,
        matches=sum(c['matches'] for c in countries.values()), candidate_pairs=pairs,
        mean_candidates_per_source1=pairs / rows,
        candidates_per_source1=dict(min=min(histogram), median=quantile(histogram, rows, .5),
                                   p90=quantile(histogram, rows, .9), p95=quantile(histogram, rows, .95),
                                   p99=quantile(histogram, rows, .99), max=max(histogram),
                                   empty_rows=histogram.get(0, 0)),
        country_compatible_cartesian_pairs=country_cartesian,
        retained_fraction_of_country_compatible_pairs=pairs / country_cartesian,
        retained_fraction_of_global_cartesian_pairs=pairs / (rows * target_rows),
        countries=countries,
        limitation='Reduction fractions describe pairs fed to the matcher. Current dense retrieval computes exhaustive same-country similarities; these are not end-to-end compute reductions.',
        elapsed_seconds=time.monotonic() - started)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
