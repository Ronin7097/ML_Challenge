"""Score complete routed owner groups; preserve exact identities and missingness."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from encode import sha
from qwen_model import inputs, load, logits
from retrieve_owners import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--retrieval',type=Path,required=True)
    p.add_argument('--owner-model',type=Path,required=True);p.add_argument('--base-model',type=Path,required=True)
    p.add_argument('--adapter',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--split',choices=['train','test'],default='train');p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--max-length',type=int,default=320);p.add_argument('--max-groups',type=int,default=0)
    p.add_argument('--reuse-selection',type=Path,help='Use an earlier score run selection for a paired model comparison')
    p.add_argument('--route-min',type=float,default=.05)
    p.add_argument('--route-max',type=float,default=.995)
    p.add_argument('--route-close-min',type=float,default=.02)
    p.add_argument('--route-close-margin',type=float,default=.5)
    p.add_argument('--restrict-roles',nargs='*',default=[],
                   help='Train-only scoring subset for fitting/evaluation; omitted for production')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if not 0 <= a.route_min < a.route_max <= 1 or not 0 <= a.route_close_min <= 1 or not 0 <= a.route_close_margin <= 1:
        raise ValueError('Invalid label-free routing thresholds')
    if a.restrict_roles and a.split!='train':
        raise ValueError('Role restriction is only valid on labelled training partitions')
    torch.set_num_threads(6);torch.cuda.set_per_process_memory_fraction(.5)
    tokenizer,model=load(a.base_model,a.adapter);model.eval()
    weights={p.name:sha(p) for p in sorted(a.base_model.glob('*.safetensors'))}
    if not weights:raise ValueError('No base model weights to identify')
    adapter_weights=sha(a.adapter/'adapter_model.safetensors') if a.adapter else None
    for source in sorted(a.data.glob(f'{a.split}_s1_*.parquet')):
        country=source.stem.removeprefix(f'{a.split}_s1_');target=a.data/f'{a.split}_targets_{country}.parquet'
        retrieval=a.retrieval/f'{a.split}_{country}'
        complete=json.loads((retrieval/'complete.json').read_text());k=complete['signature']['k']
        if complete['signature']['source_sha256']!=sha(source) or complete['signature']['target_sha256']!=sha(target):
            raise ValueError('Candidate identities changed')
        dest=a.output/f'{a.split}_{country}';dest.mkdir(exist_ok=True)
        winner_file=a.owner_model/f'all_winners_{country}.parquet'
        signature={'base_weights':weights,'adapter_sha256':adapter_weights,'max_length':a.max_length,
                   'max_groups':a.max_groups,'retrieval':complete['signature'],'winner_sha256':sha(winner_file),
                   'route':{'min':a.route_min,'max':a.route_max,'close_min':a.route_close_min,
                            'close_margin':a.route_close_margin,'restrict_roles':a.restrict_roles},
                   'prompt_source_sha256':sha(Path(__file__).with_name('qwen_model.py'))}
        config=dest/'config.json'
        if config.exists() and json.loads(config.read_text())!=signature:raise ValueError('Score cache configuration differs')
        atomic_json(config,signature)
        candidates=np.load(retrieval/'indices.npy',mmap_mode='r')
        if a.reuse_selection:
            selection=pq.read_table(a.reuse_selection/f'{a.split}_{country}'/'selection.parquet')
            selected=selection['target_row'].to_numpy()
            saved=np.asarray(selection['owner_rows'].to_pylist(),dtype=np.int32)
            if not np.array_equal(saved,candidates[selected]):raise ValueError('Paired model comparison has different owner groups')
        else:
            table=pq.read_table(winner_file,columns=['target_row','pair_probability','pair_margin'])
            rows=table['target_row'].to_numpy();prob=table['pair_probability'].to_numpy();margin=table['pair_margin'].to_numpy()
            route=((prob>a.route_min)&(prob<a.route_max))|((prob>a.route_close_min)&(margin<a.route_close_margin))
            selected=rows[route]
            if a.restrict_roles:
                source_roles=np.asarray(pq.read_table(source,columns=['role'])['role'].to_pylist())
                allowed=np.isin(source_roles,a.restrict_roles)
                selected=selected[allowed[candidates[selected]].any(axis=1)]
            if a.max_groups and len(selected)>a.max_groups:
                selected=np.random.default_rng(20260927).choice(selected,a.max_groups,replace=False)
            selected=np.sort(selected)
            selection=pa.table({'target_row':selected,'owner_rows':np.asarray(candidates[selected]).tolist()})
        if len(np.unique(selected))!=len(selected):raise ValueError('Duplicate target group')
        selection_path=dest/'selection.parquet'
        if selection_path.exists():
            if not pq.read_table(selection_path).equals(selection):raise ValueError('Changed route in existing output')
        else:pq.write_table(selection,selection_path,compression='zstd')
        if (dest/'complete.json').exists():continue
        sources=pq.read_table(source,columns=['text'])['text'];targets=pq.read_table(target,columns=['text'])['text']
        started=time.monotonic();done=0
        for start in range(0,len(selected),256):
            rows=selected[start:start+256];owners=np.asarray(candidates[rows])
            output=dest/f'scores-{start//256:05d}.npz'
            if output.exists():
                with np.load(output) as old:
                    if not np.array_equal(old['target_rows'],rows) or not np.array_equal(old['owner_rows'],owners) or not np.isfinite(old['scores']).all():
                        raise ValueError('Score shard identities or values differ')
                done+=len(rows);continue
            flat_rows=np.repeat(rows,k);flat_owners=owners.reshape(-1);scores=[]
            with torch.inference_mode():
                for begin in range(0,len(flat_rows),a.batch_size):
                    end=begin+a.batch_size
                    target_text=targets.take(pa.array(flat_rows[begin:end])).to_pylist()
                    source_text=sources.take(pa.array(flat_owners[begin:end])).to_pylist()
                    batch=inputs(tokenizer,target_text,source_text,a.max_length)
                    scores.append(logits(tokenizer,model,batch).softmax(dim=1)[:,1].cpu().numpy())
            values=np.concatenate(scores).reshape(-1,k)
            if not np.isfinite(values).all():raise FloatingPointError('Incomplete/nonfinite reranker group')
            temp=output.with_suffix('.npz.partial')
            with temp.open('wb') as f:np.savez_compressed(f,target_rows=rows,owner_rows=owners,scores=values)
            temp.replace(output);done+=len(rows)
            print(json.dumps({'country':country,'groups_scored':done,'groups_total':len(selected),'seconds':time.monotonic()-started}),flush=True)
        atomic_json(dest/'complete.json',{'groups':len(selected),'pairs':len(selected)*k,'complete_groups':True,
            'bounded_smoke':bool(a.max_groups),'seconds':time.monotonic()-started,'selection_sha256':sha(selection_path)})


if __name__=='__main__':main()
