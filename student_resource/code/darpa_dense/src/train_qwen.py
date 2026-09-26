"""Fine-tune a local Qwen reranker adapter on supplied fitting records only."""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from peft import LoraConfig, get_peft_model
from transformers import get_linear_schedule_with_warmup

from encode import sha
from qwen_model import MODEL, REVISION, inputs, load, logits
from retrieve_owners import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pairs',type=Path,required=True);p.add_argument('--base-model',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--accumulate',type=int,default=4);p.add_argument('--max-length',type=int,default=320)
    p.add_argument('--max-steps',type=int,default=0);p.add_argument('--learning-rate',type=float,default=2e-5)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/'config.json').exists():raise ValueError('Use a fresh output; interrupted adapter fits are not silently reused')
    torch.set_num_threads(6);torch.cuda.set_per_process_memory_fraction(.5);torch.manual_seed(20260927)
    tokenizer,base=load(a.base_model,training=True)
    model=get_peft_model(base,LoraConfig(r=16,lora_alpha=32,lora_dropout=.05,bias='none',task_type='CAUSAL_LM',
        target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.train()
    data=pq.read_table(a.pairs).to_pydict();n=len(data['label'])
    order=np.random.default_rng(20260927).permutation(n)
    effective=a.batch_size*a.accumulate;steps=math.ceil(n/effective)
    if a.max_steps:steps=min(steps,a.max_steps)
    if steps<1:raise ValueError('No training steps')
    config={'base_model':MODEL,'revision':REVISION,'pairs_sha256':sha(a.pairs),'rows':n,'steps':steps,
        'batch_size':a.batch_size,'accumulate':a.accumulate,'max_length':a.max_length,
        'learning_rate':a.learning_rate,'seed':20260927,'lora_rank':16,'lora_alpha':32,
        'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad)}
    atomic_json(a.output/'config.json',config)
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=a.learning_rate,weight_decay=.01)
    scheduler=get_linear_schedule_with_warmup(optimizer,max(1,steps//20),steps)
    started=time.monotonic()
    for step in range(steps):
        ids=order[step*effective:(step+1)*effective];optimizer.zero_grad(set_to_none=True);loss_sum=0.
        for start in range(0,len(ids),a.batch_size):
            selected=ids[start:start+a.batch_size]
            batch=inputs(tokenizer,[data['target_text'][i] for i in selected],[data['source_text'][i] for i in selected],a.max_length)
            target=torch.tensor([data['label'][i] for i in selected],device='cuda')
            loss=torch.nn.functional.cross_entropy(logits(tokenizer,model,batch),target)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite reranker loss')
            (loss*len(selected)/len(ids)).backward();loss_sum+=float(loss.detach())*len(selected)/len(ids)
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(norm):raise FloatingPointError('Nonfinite reranker gradient')
        optimizer.step();scheduler.step()
        if step==0 or (step+1)%20==0:
            report={'step':step+1,'total_steps':steps,'loss':loss_sum,'seconds':time.monotonic()-started}
            atomic_json(a.output/'progress.json',report);print(json.dumps(report),flush=True)
        if (step+1)%250==0:model.save_pretrained(a.output/f'adapter_step_{step+1:05d}')
    model.save_pretrained(a.output/'adapter');tokenizer.save_pretrained(a.output/'tokenizer')
    atomic_json(a.output/'complete.json',{**config,'steps_completed':steps,'elapsed_seconds':time.monotonic()-started})


if __name__=='__main__':main()
