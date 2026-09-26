"""End-to-end synthetic integration check; never a challenge accuracy estimate."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from pair_features import FEATURE_NAMES, shortlist


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path);a=p.parse_args()
    data=a.output/'data';features=a.output/'features/train_India'
    if a.output.exists():raise ValueError('Use a new smoke output directory')
    data.mkdir(parents=True);features.mkdir(parents=True)
    roles=['encoder','pair','context','tune','reserve']*4
    ids=[f'S1-{i:05d}' for i in range(20)]
    names=[f'Company alpha{i:03d}' for i in range(20)]
    addresses=[f'{100+i} Cedar road Delhi {110000+i}' for i in range(20)]
    pq.write_table(pa.table({'entity_id':ids,'role':roles}),data/'train_s1_India.parquet')
    target_ids=[];target_rows=[];owner_rows=[];matrices=[];truth=[]
    for i in range(20):
        for j in range(200):
            row=len(target_ids);tid=f'S2-{row:06d}';target_ids.append(tid)
            owners=np.array([(i+k)%20 for k in range(10)])
            matrix=shortlist([(names[k],addresses[k]) for k in owners],(names[i],addresses[i]),np.linspace(.99,.5,10),np.ones(10),False)
            matrices.append(matrix);target_rows.extend([row]*10);owner_rows.extend(owners)
            if j<160:truth.append((ids[i],tid))
    pq.write_table(pa.table({'entity_id':target_ids}),data/'train_targets_India.parquet')
    db=duckdb.connect(str(data/'records.duckdb'))
    db.execute('CREATE TABLE owners(s1_id VARCHAR,target_id VARCHAR)');db.executemany('INSERT INTO owners VALUES (?,?)',truth);db.close()
    matrix=np.concatenate(matrices)
    pq.write_table(pa.table({'target_row':target_rows,'owner_row':owner_rows,**{n:matrix[:,k] for k,n in enumerate(FEATURE_NAMES)}}),features/'features-00000.parquet')
    (features/'config.json').write_text(json.dumps({'columns':FEATURE_NAMES,'roles':['pair','context','tune'],'retrieval':{'k':10}}))
    (features/'complete.json').write_text(json.dumps({'full_competitor_shortlists':True}))
    subprocess.run([sys.executable,str(Path(__file__).with_name('fit_owner.py')),'--data',str(data),'--features',str(a.output/'features'),'--output',str(a.output/'model'),'--bags','1'],check=True)
    report=json.loads((a.output/'model/complete.json').read_text())
    assert report['reserve_evaluated'] is False
    lines=(a.output/'model/tune_matching_results.tsv').read_text().splitlines()
    assert len(lines)==roles.count('tune')+1
    assert set(line.split('\t')[0] for line in lines[1:])=={ids[i] for i,r in enumerate(roles) if r=='tune'}
    print('Synthetic complete-owner pipeline passed; result is not a challenge metric.',flush=True)


if __name__=='__main__':main()
