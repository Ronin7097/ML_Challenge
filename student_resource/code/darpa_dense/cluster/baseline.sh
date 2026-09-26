#!/usr/bin/env bash
set -euo pipefail
umask 077
export TMPDIR="$PWD/tmp"
mkdir -p "$TMPDIR" bin work/baseline
TMPDIR="$PWD/tmp" g++ -O3 -std=c++17 -pthread -I ../student_resource/code/business_entity_resolution/src src/baseline_probe.cpp -o bin/baseline_probe
.venv/bin/python - <<'PREP'
import duckdb,csv
from pathlib import Path
db=duckdb.connect('work/data/records.duckdb',read_only=True)
Path('work/baseline/tune_ids.txt').write_text(''.join(r[0]+'\n' for r in db.execute("SELECT entity_id FROM roles WHERE role='tune' ORDER BY entity_id").fetchall()))
with open('../student_resource/dataset/test/test_source1.tsv') as f:
 rows=csv.DictReader(f,delimiter='\t')
 Path('work/baseline/parity_ids.txt').write_text(''.join(next(rows)['entity_id']+'\n' for _ in range(8)))
PREP
if [[ ! -f work/baseline/parity/complete.json ]]; then
bin/baseline_probe ../student_resource/dataset test ../student_resource/code/business_entity_resolution/models work/baseline/parity_ids.txt work/baseline/parity 4
fi
.venv/bin/python - <<'VERIFY'
import csv,json
from pathlib import Path
for name in ('matching_results.tsv','candidate_pairs.tsv'):
 with open('work/baseline/parity/'+name) as f: expected=dict(list(csv.reader(f,delimiter='\t'))[1:])
 with open('../output_advanced/'+name) as f:
  reader=csv.reader(f,delimiter='\t');next(reader)
  for q,ids in reader:
   if q in expected:
    assert ids==expected.pop(q), (name,q,'frozen prediction mismatch')
   if not expected:break
 assert not expected
Path('work/baseline/parity_verified.json').write_text(json.dumps({'queries':8,'matching_identical':True,'candidate_identical':True})+'\n')
print('Frozen subset parity passed',flush=True)
VERIFY
if [[ ! -f work/baseline/tune/complete.json ]]; then
bin/baseline_probe ../student_resource/dataset train ../student_resource/code/business_entity_resolution/models work/baseline/tune_ids.txt work/baseline/tune 6
fi

.venv/bin/python src/evaluate_tune.py --data work/data --prediction work/baseline/tune/matching_results.tsv --output work/baseline/tune_report.json
