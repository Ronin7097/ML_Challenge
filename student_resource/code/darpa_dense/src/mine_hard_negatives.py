"""Mine fitting-only wrong owners without processing every training target."""
import argparse
import json
import time
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from encode import sha
from pair_features import normalized
from prepare import sql
from retrieve_owners import atomic_json
from train_encoder import embed, load


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--vectors',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--owners-per-country',type=int,default=50000);p.add_argument('--batch-size',type=int,default=128)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/'complete.json').exists():raise ValueError('Mining already complete')
    torch.set_num_threads(6);torch.cuda.set_per_process_memory_fraction(.5)
    tokenizer,model=load(a.model);model.eval();model_hash=sha(a.model/'model.safetensors')
    db=duckdb.connect();db.execute("SET threads=6; SET memory_limit='12GB'")
    output=[];reports={};started=time.monotonic()
    for source in sorted(a.data.glob('train_s1_*.parquet')):
        country=source.stem.removeprefix('train_s1_')
        meta=json.loads((a.vectors/(source.stem+'.json')).read_text())
        if meta['model_sha256']!=model_hash or meta['input_sha256']!=sha(source) or meta['max_length']!=96:
            raise ValueError('Reference vectors differ from mining model or data')
        sources=pq.read_table(source,columns=['entity_id','text','role','name_group']).to_pydict()
        source_index={v:i for i,v in enumerate(sources['entity_id'])}
        rows=db.execute(f"SELECT * FROM read_parquet({sql(a.data/'encoder_pairs.parquet')}) WHERE country={sql(country)} ORDER BY md5(s1_id || 'qwen20260927') LIMIT {a.owners_per_country}").fetchall()
        raw=np.load(a.vectors/(source.stem+'.npy'),mmap_mode='r')
        reference=F.normalize(torch.tensor(np.asarray(raw),device='cuda',dtype=torch.float16),dim=1)
        kept=0;identical_excluded=0
        with torch.inference_mode():
            for start in range(0,len(rows),a.batch_size):
                batch=rows[start:start+a.batch_size]
                tokens=tokenizer([r[5] for r in batch],padding=True,truncation=True,max_length=96,return_tensors='pt').to('cuda')
                query=F.normalize(embed(model,tokens).half(),dim=1)
                _,indices=torch.topk(query@reference.T,k=32,dim=1,sorted=True)
                for row,choices in zip(batch,indices.cpu().numpy()):
                    owner,tid,c,group,source_text,target_text=row;own=source_index[owner]
                    if sources['role'][own]!='encoder':raise ValueError('Held-out fitting owner')
                    rival=None
                    for candidate in choices:
                        candidate=int(candidate)
                        if candidate==own or sources['role'][candidate]!='encoder':continue
                        if normalized(sources['text'][candidate])==normalized(source_text):
                            identical_excluded+=1;continue
                        rival=candidate;break
                    if rival is None:continue
                    for label,idx in ((1,own),(0,rival)):
                        output.append({'true_owner':owner,'target_id':tid,'candidate_owner':sources['entity_id'][idx],
                            'country':c,'target_text':target_text,'source_text':sources['text'][idx],'label':label,
                            'same_name_group':sources['name_group'][idx]==sources['name_group'][own]})
                    kept+=1
                if start%(a.batch_size*50)==0:
                    print(json.dumps({'country':country,'mined_owners':kept,'scanned':start+len(batch),'total':len(rows)}),flush=True)
        reports[country]={'owners_requested':len(rows),'owners_mined':kept,'indistinguishable_negatives_skipped':identical_excluded}
        del reference;torch.cuda.empty_cache()
    path=a.output/'pairs.parquet';pq.write_table(pa.Table.from_pylist(output),path,compression='zstd')
    report={'rows':len(output),'positive_pairs':sum(r['label'] for r in output),
            'same_name_negative_pairs':sum(r['same_name_group'] and not r['label'] for r in output),
            'fit_role':'encoder only, including negative owners','mining_model_sha256':model_hash,
            'pairs_sha256':sha(path),'countries':reports,'reserve_evaluated':False,'seconds':time.monotonic()-started}
    atomic_json(a.output/'complete.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
