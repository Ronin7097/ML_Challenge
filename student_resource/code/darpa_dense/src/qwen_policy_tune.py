"""Compare bounded Qwen owner overrides with the frozen owner/context tune policy."""
import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pyarrow.parquet as pq

from encode import sha
from fit_owner import ROLE, role_metadata
from retrieve_owners import atomic_json


def load_scored(folder):
    complete=json.loads((folder/'complete.json').read_text())
    if not complete['complete_groups'] or complete['bounded_smoke']:
        raise ValueError('Need complete unbounded routed groups')
    selection=folder/'selection.parquet'
    if sha(selection)!=complete['selection_sha256']:
        raise ValueError('Qwen selected-group identity changed')
    rows=[];owners=[];scores=[]
    for file in sorted(folder.glob('scores-*.npz')):
        with np.load(file) as chunk:
            rows.append(chunk['target_rows']);owners.append(chunk['owner_rows']);scores.append(chunk['scores'])
    tr,ow,p=map(np.concatenate,(rows,owners,scores))
    selected=pq.read_table(selection)
    if len(tr)!=complete['groups'] or not np.array_equal(tr,selected['target_row'].to_numpy()):
        raise ValueError('Qwen shards do not cover the selected groups exactly')
    saved=np.asarray(selected['owner_rows'].to_pylist(),dtype=np.int32)
    if not np.array_equal(ow,saved) or not np.isfinite(p).all():
        raise ValueError('Qwen shard owners/scores differ from selection')
    index=p.argmax(axis=1)
    ranked=np.sort(p,axis=1)
    return tr,ow[np.arange(len(tr)),index],p[np.arange(len(tr)),index],ranked[:,-1]-ranked[:,-2]


def macro_f05(choices,metadata):
    total=count=tp_count=fp_count=fn_count=0
    for country,by_target in choices.items():
        ids,roles,gold,_,_=metadata[country]
        tr=np.fromiter(by_target.keys(),dtype=np.int64,count=len(by_target))
        ow=np.fromiter(by_target.values(),dtype=np.int32,count=len(by_target))
        predicted=np.bincount(ow,minlength=len(ids))
        tp=np.bincount(ow,weights=gold[tr]==ow,minlength=len(ids))
        truth=np.bincount(gold[gold>=0],minlength=len(ids))
        den=.25*truth+predicted
        scores=np.divide(1.25*tp,den,out=np.ones(len(ids)),where=den!=0)
        mask=roles==ROLE['tune']
        total+=scores[mask].sum();count+=int(mask.sum())
        tp_count+=int(tp[mask].sum())
        fp_count+=int((predicted[mask]-tp[mask]).sum())
        fn_count+=int((truth[mask]-tp[mask]).sum())
    return float(total/count),{'tp':tp_count,'fp':fp_count,'fn':fn_count}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--owner-model',type=Path,required=True)
    parser.add_argument('--qwen-scores',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--base-threshold',type=float,default=.62)
    a=parser.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/'complete.json').exists():
        raise ValueError('Policy already evaluated; use a new output')
    db=duckdb.connect(str(a.data/'records.duckdb'),read_only=True)
    metadata={};base={};routed={};identity={}
    for source in sorted(a.data.glob('train_s1_*.parquet')):
        country=source.stem.removeprefix('train_s1_')
        target=a.data/f'train_targets_{country}.parquet'
        metadata[country]=(*role_metadata(db,source,target),None)
        ids,roles,_,_,_=metadata[country]
        decisions=a.owner_model/f'decisions_{country}.parquet'
        table=pq.read_table(decisions)
        tr=table['target_row'].to_numpy();ow=table['owner_row'].to_numpy();p=table['context_probability'].to_numpy()
        selected=(p>=a.base_threshold)&(roles[ow]==ROLE['tune'])
        base[country]={int(t):int(o) for t,o in zip(tr[selected],ow[selected])}
        if len(base[country])!=int(selected.sum()):
            raise ValueError('Duplicate original target decision')
        folder=a.qwen_scores/f'train_{country}'
        qtr,qow,qp,qmargin=load_scored(folder)
        routed[country]=(qtr,qow,qp,qmargin)
        identity[country]={'decision_sha256':sha(decisions),
                           'qwen_selection_sha256':sha(folder/'selection.parquet'),
                           'qwen_groups':len(qtr),
                           'score_config':json.loads((folder/'config.json').read_text())}
    baseline_score,baseline_links=macro_f05(base,metadata)
    old=json.loads((a.owner_model/'complete.json').read_text())['best_tune']['macro_f05']
    if abs(baseline_score-old)>1e-10:
        raise ValueError(f'Frozen base score does not reproduce: {baseline_score} versus {old}')
    sweep=[]
    for threshold in (.2,.5,.7,.9,.95,.99,.995,.999):
        for margin in (0.,.05,.1,.2,.3):
            for reject_low in (False,True):
                choices={country:by_target.copy() for country,by_target in base.items()}
                for country,(tr,ow,score,gap) in routed.items():
                    roles=metadata[country][1]
                    current=choices[country]
                    strong=(score>=threshold)&(gap>=margin)
                    for i,t in enumerate(tr):
                        target=int(t)
                        if strong[i]:
                            if roles[ow[i]]==ROLE['tune']:
                                current[target]=int(ow[i])
                            else:
                                current.pop(target,None)
                        elif reject_low:
                            current.pop(target,None)
                f05,links=macro_f05(choices,metadata)
                sweep.append({'qwen_threshold':threshold,'qwen_margin':margin,
                              'reject_low':reject_low,'macro_f05':f05,'links':links})
    best=max(sweep,key=lambda row:(row['macro_f05'],row['qwen_threshold'],row['qwen_margin'],row['reject_low']))
    choices={country:by_target.copy() for country,by_target in base.items()}
    for country,(tr,ow,score,gap) in routed.items():
        roles=metadata[country][1];current=choices[country]
        for i,t in enumerate(tr):
            target=int(t)
            if score[i]>=best['qwen_threshold'] and gap[i]>=best['qwen_margin']:
                if roles[ow[i]]==ROLE['tune']:current[target]=int(ow[i])
                else:current.pop(target,None)
            elif best['reject_low']:current.pop(target,None)
    output=a.output/'tune_matching_results.tsv'
    with output.open('w') as f:
        f.write('source1_entity_id\tmatched_entity_ids\n')
        for country,current in choices.items():
            ids,roles,_,_,_=metadata[country]
            target_ids=pq.read_table(a.data/f'train_targets_{country}.parquet',columns=['entity_id'])['entity_id'].to_pylist()
            grouped={}
            for tr,ow in current.items():
                grouped.setdefault(ow,[]).append(target_ids[tr])
            for ow in np.flatnonzero(roles==ROLE['tune']):
                f.write(ids[ow]+'\t'+','.join(sorted(grouped.get(int(ow),[])))+'\n')
    atomic_json(a.output/'complete.json',{'best_tune':best,'base_threshold':a.base_threshold,
        'frozen':False,'base_tune_f05':baseline_score,
        'base_links':baseline_links,'sweep':sweep,'identities':identity,'reserve_opened':False,
        'warning':'Thresholds selected on tune; independent reserve evaluation still required'})
    print(json.dumps({'base':baseline_score,'best':best}),flush=True)


if __name__=='__main__':main()
