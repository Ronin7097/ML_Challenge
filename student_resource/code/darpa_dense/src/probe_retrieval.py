"""Measure true-owner retrieval on tune links; this does NOT estimate macro F0.5."""
import argparse
import json
import time
from pathlib import Path

import duckdb
import numpy as np
import pyarrow.parquet as pq
import torch

from prepare import sql
from train_encoder import embed, load


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data",type=Path,required=True)
    p.add_argument("--vectors",type=Path,required=True)
    p.add_argument("--model",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--batch-size",type=int,default=128)
    p.add_argument("--max-length",type=int,default=96)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/"report.json").exists():
        raise SystemExit("Probe already complete")
    db=duckdb.connect(str(a.data/"records.duckdb"),read_only=True)
    torch.set_num_threads(6)
    torch.cuda.set_per_process_memory_fraction(0.65)
    tokenizer,model=load(a.model);model.eval()
    reports={};started=time.monotonic()
    # Only tune labels are opened; the reserve is never queried by this command.
    for country, in db.execute("SELECT DISTINCT country FROM roles WHERE role='tune' ORDER BY 1").fetchall():
        rows=db.execute(f"""SELECT o.s1_id,o.target_id,
          t.business_name || ' | ' || t.business_address || ' | ' || t.country AS text
          FROM owners o JOIN roles r ON o.s1_id=r.entity_id
          JOIN train_targets t ON o.target_id=t.entity_id
          WHERE r.role='tune' AND r.country={sql(country)} ORDER BY o.target_id""").fetchall()
        source_ids=pq.read_table(a.data/f"train_s1_{country}.parquet",columns=["entity_id"]).column(0).to_pylist()
        owner_index={name:i for i,name in enumerate(source_ids)}
        vectors=np.load(a.vectors/f"train_s1_{country}.npy",mmap_mode="r")
        reference=torch.as_tensor(np.asarray(vectors).copy(),device="cuda",dtype=torch.float16)
        reference=torch.nn.functional.normalize(reference,dim=1)
        counters={k:0 for k in (1,2,4,8,10,20,32)}
        examples=[]
        with torch.inference_mode():
            for start in range(0,len(rows),a.batch_size):
                batch=rows[start:start+a.batch_size]
                tokens=tokenizer([r[2] for r in batch],padding=True,truncation=True,max_length=a.max_length,return_tensors="pt").to("cuda")
                query=embed(model,tokens).to(torch.float16)
                # Exhaustive GPU inner products avoid approximate-index recall confounding.
                _,indices=torch.topk(query@reference.T,k=32,dim=1,sorted=True)
                choices=indices.cpu().numpy()
                gold=np.asarray([owner_index[r[0]] for r in batch])
                for k in counters:
                    counters[k]+=int((choices[:,:k]==gold[:,None]).any(axis=1).sum())
                for j,(s1,tid,_) in enumerate(batch):
                    if gold[j] not in choices[j] and len(examples)<20:
                        examples.append({"source1":s1,"target":tid,"top_owner":source_ids[int(choices[j,0])]})
                if start%(a.batch_size*50)==0:
                    print(json.dumps({"country":country,"links_checked":start+len(batch),"links_total":len(rows)}),flush=True)
        reports[country]={"true_links":len(rows),"owner_pool":len(source_ids),"retrieved":counters,
                          "recall":{k:v/len(rows) for k,v in counters.items()},"miss_examples_tune_only":examples}
        del reference
        torch.cuda.empty_cache()
    total=sum(v["true_links"] for v in reports.values())
    result={"evaluation":"true-owner candidate recall on tune links; no F0.5 claim",
            "reserve_opened":False,"countries":reports,"true_links":total,
            "recall":{k:sum(r["retrieved"][k] for r in reports.values())/total for k in counters},
            "elapsed_seconds":time.monotonic()-started}
    (a.output/"report.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result),flush=True)


if __name__=="__main__":
    main()
