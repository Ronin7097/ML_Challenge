"""HN1: continue a control encoder with label-confirmed retrieved wrong owners."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import get_linear_schedule_with_warmup

from encode import sha
from pair_features import normalized
from retrieve_owners import atomic_json
from train_encoder import gradient_cache, load


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pairs',type=Path,required=True);p.add_argument('--base-model',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--batch-size',type=int,default=256)
    p.add_argument('--microbatch',type=int,default=64);p.add_argument('--max-length',type=int,default=96)
    p.add_argument('--learning-rate',type=float,default=1e-5);p.add_argument('--max-steps',type=int,default=0)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/'config.json').exists():raise ValueError('Use a fresh HN1 experiment output')
    raw=pq.read_table(a.pairs).to_pylist();positive={};negative={}
    for row in raw:
        (positive if row['label'] else negative)[row['true_owner']]=row
    rows=[]
    for owner,pos in positive.items():
        if owner not in negative:continue
        neg=negative[owner]
        if pos['candidate_owner']!=owner or neg['candidate_owner']==owner or pos['target_id']!=neg['target_id']:
            raise ValueError('Mined pair ownership mismatch')
        if normalized(pos['source_text'])==normalized(neg['source_text']):
            continue
        rows.append((pos,neg))
    if not rows:raise ValueError('No distinct label-confirmed hard negatives')
    rng=np.random.default_rng(20260927);schedule=[]
    for country in sorted({p['country'] for p,n in rows}):
        ids=rng.permutation([i for i,(p,n) in enumerate(rows) if p['country']==country])
        schedule.extend([ids[i:i+a.batch_size] for i in range(0,len(ids),a.batch_size) if len(ids[i:i+a.batch_size])>1])
    schedule=[schedule[i] for i in rng.permutation(len(schedule))]
    if a.max_steps:schedule=schedule[:a.max_steps]
    torch.set_num_threads(6);torch.cuda.set_per_process_memory_fraction(.5);torch.manual_seed(20260927)
    tokenizer,model=load(a.base_model);model.train()
    config={'base_weights_sha256':sha(a.base_model/'model.safetensors'),'pairs_sha256':sha(a.pairs),
            'owners':len(rows),'steps':len(schedule),'learning_rate':a.learning_rate,'max_length':a.max_length,
            'batch_size':a.batch_size,'microbatch':a.microbatch,'same_name_mask':False,
            'duplicate_owner_and_serialized_text_masks':True,'reserve_evaluated':False}
    atomic_json(a.output/'config.json',config)
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.learning_rate,weight_decay=.01)
    scheduler=get_linear_schedule_with_warmup(optimizer,max(1,len(schedule)//20),len(schedule));started=time.monotonic()
    for step,idx in enumerate(schedule):
        chosen=[rows[i] for i in idx];n=len(chosen)
        target=[p['target_text'] for p,neg in chosen]
        source=[p['source_text'] for p,neg in chosen]+[neg['source_text'] for p,neg in chosen]
        owners=np.asarray([p['true_owner'] for p,neg in chosen])
        candidate_owners=np.r_[owners,[neg['candidate_owner'] for p,neg in chosen]]
        compact=np.asarray([normalized(t) for t in source])
        mask=(owners[:,None]==candidate_owners[None,:]) | (compact[:n,None]==compact[None,:])
        target_compact=np.asarray([normalized(t) for t in target])
        duplicate_target=target_compact[:,None]==target_compact[None,:]
        mask |= (duplicate_target.astype(np.int32) @ (owners[:,None]==candidate_owners[None,:]).astype(np.int32))>0
        mask[np.arange(n),np.arange(n)]=False
        tokens=[tokenizer(text,padding=True,truncation=True,max_length=a.max_length,return_tensors='pt').to('cuda') for text in (target,source)]
        optimizer.zero_grad(set_to_none=True)
        loss=gradient_cache(model,tokens,a.microbatch,torch.tensor(mask,device='cuda'))
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(norm):raise FloatingPointError('Nonfinite mined-encoder gradient')
        optimizer.step();scheduler.step()
        if step==0 or (step+1)%20==0:
            report={'step':step+1,'total_steps':len(schedule),'loss':loss,'seconds':time.monotonic()-started}
            atomic_json(a.output/'progress.json',report);print(json.dumps(report),flush=True)
    model.save_pretrained(a.output/'model');tokenizer.save_pretrained(a.output/'model')
    atomic_json(a.output/'complete.json',{**config,'steps_completed':len(schedule),'elapsed_seconds':time.monotonic()-started})


if __name__=='__main__':main()
