"""Build Qwen fitting pairs from encoder-role owners and retrieved hard negatives."""
import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from encode import sha
from prepare import sql
from pair_features import normalized
from retrieve_owners import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--retrieval',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--owners-per-country',type=int,default=50000)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/'complete.json').exists():raise ValueError('Qwen fitting data already complete')
    db=duckdb.connect(str(a.data/'records.duckdb'),read_only=True)
    rows=[];provenance={}
    for source in sorted(a.data.glob('train_s1_*.parquet')):
        country=source.stem.removeprefix('train_s1_');target=a.data/f'train_targets_{country}.parquet'
        folder=a.retrieval/f'train_{country}'
        meta=json.loads((folder/'complete.json').read_text())
        if meta['signature']['source_sha256']!=sha(source) or meta['signature']['target_sha256']!=sha(target):
            raise ValueError('Retrieval identities differ from current records')
        provenance[country]=meta
        s=pq.read_table(source,columns=['entity_id','text','role','name_group']).to_pydict()
        index={v:i for i,v in enumerate(s['entity_id'])}
        choices=np.load(folder/'indices.npy',mmap_mode='r')
        selected=db.execute(f'''WITH targets AS (
            SELECT entity_id,row_number() OVER(ORDER BY entity_id)-1 AS pos FROM read_parquet({sql(target)}))
            SELECT p.*,t.pos FROM read_parquet({sql(a.data/'encoder_pairs.parquet')}) p
            JOIN targets t ON p.target_id=t.entity_id WHERE p.country={sql(country)}
            ORDER BY md5(p.s1_id || 'qwen20260927') LIMIT {a.owners_per_country}''').fetchall()
        for owner,tid,c,group,source_text,target_text,position in selected:
            own=index[owner]
            if s['role'][own]!='encoder':raise ValueError('Held-out owner entered reranker fitting')
            rivals=[int(i) for i in choices[position] if int(i)!=own and s['role'][int(i)]=='encoder'
                    and normalized(s['text'][int(i)])!=normalized(source_text)]
            if not rivals:continue
            for label,candidate in ((1,own),(0,rivals[0])):
                rows.append({'true_owner':owner,'target_id':tid,'candidate_owner':s['entity_id'][candidate],
                             'country':c,'target_text':target_text,'source_text':s['text'][candidate], 'label':label,
                             'same_name_group':s['name_group'][candidate]==s['name_group'][own]})
    if not rows:raise ValueError('No reranker pairs available')
    path=a.output/'pairs.parquet';pq.write_table(pa.Table.from_pylist(rows),path,compression='zstd')
    atomic_json(a.output/'complete.json',{'rows':len(rows),'positive':sum(r['label'] for r in rows),
        'fit_role':'encoder only, including hard-negative owners','pairs_sha256':sha(path),
        'same_name_negative_pairs':sum(r['same_name_group'] and not r['label'] for r in rows),
        'retrieval':provenance,'reserve_evaluated':False})
    print(json.dumps({'reranker_pairs':len(rows)}),flush=True)


if __name__=='__main__':main()
