"""Freeze and measure a bounded rich-context route on existing tune scores."""
import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pyarrow.parquet as pq

from encode import sha
from fit_owner import ROLE, role_metadata
from fit_rich_context import metrics
from retrieve_owners import atomic_json


def effective_probability(base, rich, base_threshold, route_min, route_max):
    route=(base>route_min)&(base<route_max)
    return np.where(route,rich,(base>=base_threshold).astype(float)),route


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--rich',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--route-min',type=float,default=.02)
    p.add_argument('--route-max',type=float,default=.999)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/'policy.json').exists():raise ValueError('Use a new policy output')
    frozen=json.loads((a.base/'complete.json').read_text());base_threshold=frozen['best_tune']['threshold']
    db=duckdb.connect(str(a.data/'records.duckdb'),read_only=True)
    data={};identity={}
    for source in sorted(a.data.glob('train_s1_*.parquet')):
        c=source.stem.removeprefix('train_s1_')
        m=role_metadata(db,source,a.data/f'train_targets_{c}.parquet')
        rich=pq.read_table(a.rich/f'decisions_{c}.parquet').to_pydict()
        base=pq.read_table(a.base/f'decisions_{c}.parquet').to_pydict()
        lookup={int(tr):(int(ow),float(p)) for tr,ow,p in zip(base['target_row'],base['owner_row'],base['context_probability'])}
        tr,ow=np.asarray(rich['target_row']),np.asarray(rich['owner_row'])
        if any(lookup[int(t)][0]!=o for t,o in zip(tr,ow)):raise ValueError('Rich and base winner identity mismatch')
        bp=np.array([lookup[int(t)][1] for t in tr]);rp=np.asarray(rich['context_probability'])
        effective,route=effective_probability(bp,rp,base_threshold,a.route_min,a.route_max)
        data[c]=(m,tr,ow,effective,route)
        identity[c]={'base':sha(a.base/f'decisions_{c}.parquet'),'rich':sha(a.rich/f'decisions_{c}.parquet')}
    sweep=[]
    for threshold in np.arange(.3,.991,.01):
        values=[]
        for m,tr,ow,eff,_ in data.values():values.append(metrics(tr,ow,eff,threshold,m[1],m[2],'tune')[0])
        sweep.append({'threshold':round(float(threshold),3),'macro_f05':float(np.concatenate(values).mean())})
    best=max(sweep,key=lambda r:(r['macro_f05'],r['threshold']))
    with (a.output/'tune_matching_results.tsv').open('w') as f:
        f.write('source1_entity_id\tmatched_entity_ids\n')
        for c,(m,tr,ow,eff,_) in data.items():
            ids,roles,_,_=m;tids=pq.read_table(a.data/f'train_targets_{c}.parquet',columns=['entity_id'])['entity_id'].to_pylist()
            accept=(eff>=best['threshold'])&(roles[ow]==ROLE['tune']);grouped={}
            for ti,oi in zip(tr[accept],ow[accept]):grouped.setdefault(int(oi),[]).append(tids[int(ti)])
            for oi in np.flatnonzero(roles==ROLE['tune']):f.write(ids[oi]+'\t'+','.join(sorted(grouped.get(int(oi),[])))+'\n')
    atomic_json(a.output/'policy.json',{'frozen':True,'selected_on':'tune','base_threshold':base_threshold,
                'route_min':a.route_min,'route_max':a.route_max,'best_tune':best,'sweep':sweep,
                'base_pair_sha256':sha(a.base/'pair.txt'),'rich_complete_sha256':sha(a.rich/'complete.json'),
                'base_model_sha256':{p.name:sha(p) for p in [a.base/'pair.txt',*sorted(a.base.glob('context_*.txt'))]},
                'score_identities':identity,'reserve_used_for_selection':False})
    print(json.dumps(best),flush=True)


if __name__=='__main__':main()
