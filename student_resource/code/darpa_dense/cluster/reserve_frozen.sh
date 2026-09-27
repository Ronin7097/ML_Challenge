#!/usr/bin/env bash
set -euo pipefail
umask 077
export OMP_NUM_THREADS=6
test -f work/frozen_policy.json
.venv/bin/python - <<'PY'
import json,hashlib
from pathlib import Path
p=json.loads(Path('work/frozen_policy.json').read_text())
assert p['frozen'] and p['selected_on_tune']
assert p['selected_policy']=='owner_context' and not p['reserve_opened']
for name,digest in p['model_sha256'].items():
    actual=hashlib.sha256((Path('work/full_owner')/name).read_bytes()).hexdigest()
    assert actual==digest,name
PY
.venv/bin/python -u src/build_features.py --data work/data \
  --retrieval work/full_retrieval --output work/full_features_reserve \
  --split train --roles pair context tune reserve \
  --frozen-policy work/frozen_policy.json --workers 6
.venv/bin/python -u src/predict_frozen.py --data work/data \
  --features work/full_features_reserve --retrieval work/full_retrieval \
  --model work/full_owner --threshold .62 --split train --role reserve \
  --source1-tsv ../student_resource/dataset/train/train_source1.tsv \
  --output work/reserve_base_output
.venv/bin/python - <<'PY'
import duckdb
from pathlib import Path
db=duckdb.connect('work/data/records.duckdb',read_only=True)
ids=db.execute("SELECT entity_id FROM roles WHERE role='reserve' ORDER BY entity_id").fetchall()
assert len(ids)==7909,len(ids)
Path('work/reserve_ids.txt').write_text(''.join(row[0]+'\n' for row in ids))
PY
bin/baseline_probe ../student_resource/dataset train \
  ../student_resource/code/business_entity_resolution/models \
  work/reserve_ids.txt work/baseline/reserve 6
.venv/bin/python -u src/evaluate_tune.py --data work/data --role reserve \
  --prediction work/reserve_base_output/matching_results.tsv \
  --candidate work/reserve_base_output/candidate_pairs.tsv \
  --baseline work/baseline/reserve/matching_results.tsv \
  --output work/reserve_base_output/evaluation.json
