"""Enrich selected frozen winner rows without rescoring the candidate graph."""
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from encode import sha
from fit_owner import CONTEXT_NAMES
from retrieve_owners import atomic_json
from rich_context import EXTRA_NAMES, enrich_row, peer_indices


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--retrieval',type=Path,required=True)
    p.add_argument('--winners',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--split',choices=['train','test'],default='train')
    p.add_argument('--roles',nargs='+',default=['context','tune'])
    p.add_argument('--workers',type=int,default=6)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    for source in sorted(a.data.glob(f'{a.split}_s1_*.parquet')):
        country=source.stem.removeprefix(f'{a.split}_s1_')
        target=a.data/f'{a.split}_targets_{country}.parquet'
        winner=a.winners/f'winners_{country}.parquet'
        dest=a.output/f'{a.split}_{country}.parquet'
        manifest=dest.with_suffix('.json')
        retrieval=a.retrieval/f'{a.split}_{country}'
        signature={'source_sha256':sha(source),'target_sha256':sha(target),'winners_sha256':sha(winner),
                   'rich_code_sha256':sha(Path(__file__).with_name('rich_context.py')),
                   'builder_sha256':sha(Path(__file__)),
                   'roles':a.roles if a.split=='train' else [],'columns':CONTEXT_NAMES+EXTRA_NAMES,
                   'retrieval':json.loads((retrieval/'complete.json').read_text())['signature']}
        if signature['source_sha256']!=signature['retrieval']['source_sha256'] or signature['target_sha256']!=signature['retrieval']['target_sha256']:
            raise ValueError('Retrieval input identity mismatch')
        if manifest.exists():
            old=json.loads(manifest.read_text())
            if old['signature']!=signature or old['sha256']!=sha(dest):
                raise ValueError('Rich cache identity changed')
            continue
        started=time.monotonic()
        s=pq.read_table(source)
        t=pq.read_table(target,columns=['business_name','business_address','entity_id'])
        table=pq.read_table(winner)
        if a.split=='train':
            roles=np.asarray(s['role'].to_pylist())
            keep=np.isin(roles[table['owner_row'].to_numpy()],a.roles)
            table=table.filter(pa.array(keep))
        tr=table['target_row'].to_numpy();ow=table['owner_row'].to_numpy()
        if len(np.unique(tr))!=len(tr):raise ValueError('Duplicate winner target')
        candidates=np.load(retrieval/'indices.npy',mmap_mode='r')
        groups=np.asarray(candidates[tr]);member=groups==ow[:,None]
        if not np.all(member.sum(axis=1)==1):raise ValueError('Winner missing or duplicated in shortlist')
        rivals=np.where(groups[:,0]!=ow,groups[:,0],groups[:,1])
        ss=s.select(['business_name','business_address']).take(pa.array(ow)).to_pydict()
        rr=s.select(['business_name','business_address']).take(pa.array(rivals)).to_pydict()
        tt=t.take(pa.array(tr)).to_pydict()
        source_records=list(zip(ss['business_name'],ss['business_address']))
        rival_records=list(zip(rr['business_name'],rr['business_address']))
        target_records=list(zip(tt['business_name'],tt['business_address']))
        prob=table['pair_probability'].to_numpy()
        peers=peer_indices(ow,prob,target_records)
        rows=((source_records[i],target_records[i],rival_records[i],
               target_records[j] if j>=0 else None,float(prob[j]) if j>=0 else -1.,
               float(tt['entity_id'][i][:2]==tt['entity_id'][j][:2]) if j>=0 else 0.)
              for i,j in enumerate(peers))
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            extra=np.asarray(list(pool.map(enrich_row,rows,chunksize=1000)),dtype=np.float32)
        out=pa.table({'target_row':tr,'owner_row':ow,
                      **{n:table[n] for n in CONTEXT_NAMES},
                      **{n:extra[:,j] for j,n in enumerate(EXTRA_NAMES)}})
        temp=dest.with_suffix('.parquet.partial');pq.write_table(out,temp,compression='zstd');temp.replace(dest)
        atomic_json(manifest,{'signature':signature,'rows':len(out),'sha256':sha(dest),'seconds':time.monotonic()-started})
        print(json.dumps({'country':country,'rows':len(out),'features':len(CONTEXT_NAMES)+len(EXTRA_NAMES),
                          'seconds':time.monotonic()-started}),flush=True)


if __name__=='__main__':main()
