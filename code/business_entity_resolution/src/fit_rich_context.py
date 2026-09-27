"""Fit a richer context head on context roles; select the threshold on tune.

An internal context name-group split chooses the boosting iteration count.
No reserve outcomes enter fitting, early stopping, or threshold selection.
"""
import argparse
import hashlib
import json
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from encode import sha
from fit_owner import ROLE, CONTEXT_NAMES, role_metadata
from retrieve_owners import atomic_json
from rich_context import EXTRA_NAMES, OMIT_BASE


def metrics(tr, ow, probability, threshold, roles, gold, wanted_role):
    wanted=roles==ROLE[wanted_role]
    accepted=(probability>=threshold)&wanted[ow]
    pred=np.bincount(ow[accepted],minlength=len(roles))
    tp=np.bincount(ow[accepted],weights=gold[tr[accepted]]==ow[accepted],minlength=len(roles))
    truth=np.bincount(gold[gold>=0],minlength=len(roles))
    den=.25*truth+pred
    values=np.divide(1.25*tp,den,out=np.ones(len(roles)),where=den!=0)
    return values[wanted],{'tp':int(tp[wanted].sum()),'fp':int((pred-tp)[wanted].sum()),
                           'fn':int((truth-tp)[wanted].sum())}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--features',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--leaves',type=int,default=63)
    p.add_argument('--rounds',type=int,default=1600)
    p.add_argument('--bags',type=int,default=5)
    p.add_argument('--threads',type=int,default=6)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/'complete.json').exists():raise ValueError('Use a new model output directory')
    names=[n for n in CONTEXT_NAMES if n not in OMIT_BASE]+EXTRA_NAMES
    db=duckdb.connect(str(a.data/'records.duckdb'),read_only=True)
    metadata={};tables={};matrices={};xx=[];yy=[];vv=[];identities={}
    for source in sorted(a.data.glob('train_s1_*.parquet')):
        c=source.stem.removeprefix('train_s1_')
        path=a.features/f'train_{c}.parquet'
        manifest=json.loads(path.with_suffix('.json').read_text())
        if manifest['sha256']!=sha(path) or manifest['signature']['columns']!=CONTEXT_NAMES+EXTRA_NAMES:
            raise ValueError('Rich feature identity/schema differs')
        metadata[c]=role_metadata(db,source,a.data/f'train_targets_{c}.parquet')
        ids,roles,gold,gold_roles=metadata[c]
        table=pq.read_table(path);tables[c]=table
        tr,ow=table['target_row'].to_numpy(),table['owner_row'].to_numpy()
        if len(np.unique(tr))!=len(tr):raise ValueError('Duplicate target winners')
        x=np.column_stack([table[n].to_numpy() for n in names]).astype(np.float32);matrices[c]=x
        allowed=(roles[ow]==ROLE['context'])&np.isin(gold_roles[tr],[-1,ROLE['encoder'],ROLE['context']])
        group=pq.read_table(source,columns=['name_group'])['name_group'].to_pylist()
        validation=np.asarray([int.from_bytes(hashlib.blake2b(group[int(i)].encode(),digest_size=8).digest(),'little')%5==0 for i in ow[allowed]])
        xx.append(x[allowed]);yy.append(gold[tr[allowed]]==ow[allowed]);vv.append(validation)
        identities[c]=manifest
    x,y,val=np.concatenate(xx),np.concatenate(yy),np.concatenate(vv)
    if any(len(np.unique(y[m]))<2 for m in (val,~val)):raise ValueError('Incomplete fit/validation class coverage')
    params={'objective':'binary','metric':'binary_logloss','verbosity':-1,'num_threads':a.threads,
            'learning_rate':.035,'num_leaves':a.leaves,'min_data_in_leaf':100,'lambda_l2':10,
            'feature_fraction':.85,'bagging_fraction':.85,'bagging_freq':1,'seed':20260928,
            'deterministic':True,'force_col_wise':True}
    early=lgb.train(params,lgb.Dataset(x[~val],label=y[~val],feature_name=names),a.rounds,
        valid_sets=[lgb.Dataset(x[val],label=y[val],feature_name=names)],
        callbacks=[lgb.early_stopping(120,verbose=False)])
    rounds=early.best_iteration
    print(json.dumps({'fit_rows':len(y),'positive':int(y.sum()),'validation_rows':int(val.sum()),
                      'features':len(names),'selected_rounds':rounds}),flush=True)
    models=[]
    for seed in range(a.bags):
        model=lgb.train(dict(params,seed=20260928+seed),lgb.Dataset(x,label=y,feature_name=names),rounds)
        model.save_model(str(a.output/f'rich_{seed}.txt'));models.append(model)
        print(json.dumps({'bag_complete':seed+1,'bags':a.bags}),flush=True)
    del xx,yy,vv,x,y,val
    predictions={}
    for c,table in tables.items():
        probability=np.mean([m.predict(matrices[c],num_threads=a.threads) for m in models],axis=0)
        if not np.isfinite(probability).all():raise ValueError('Nonfinite model output')
        predictions[c]=probability
        pq.write_table(pa.table({'target_row':table['target_row'],'owner_row':table['owner_row'],
                                'context_probability':probability}),a.output/f'decisions_{c}.parquet',compression='zstd')
    sweep=[]
    for threshold in np.arange(.30,.991,.01):
        scores=[];counts={'tp':0,'fp':0,'fn':0}
        for c,probability in predictions.items():
            table=tables[c];_,roles,gold,_=metadata[c]
            values,links=metrics(table['target_row'].to_numpy(),table['owner_row'].to_numpy(),probability,threshold,roles,gold,'tune')
            scores.append(values)
            for key in counts:counts[key]+=links[key]
        sweep.append({'threshold':round(float(threshold),3),'macro_f05':float(np.concatenate(scores).mean()),'links':counts})
    best=max(sweep,key=lambda r:(r['macro_f05'],r['threshold']))
    with (a.output/'tune_matching_results.tsv').open('w') as f:
        f.write('source1_entity_id\tmatched_entity_ids\n')
        for c,probability in predictions.items():
            ids,roles,_,_=metadata[c];table=tables[c]
            tr,ow=table['target_row'].to_numpy(),table['owner_row'].to_numpy()
            tids=pq.read_table(a.data/f'train_targets_{c}.parquet',columns=['entity_id'])['entity_id'].to_pylist()
            grouped={};accept=(probability>=best['threshold'])&(roles[ow]==ROLE['tune'])
            for ti,oi in zip(tr[accept],ow[accept]):grouped.setdefault(int(oi),[]).append(tids[int(ti)])
            for oi in np.flatnonzero(roles==ROLE['tune']):f.write(ids[oi]+'\t'+','.join(sorted(grouped.get(int(oi),[])))+'\n')
    complete={'stage':'rich winner context','best_tune':best,'threshold_sweep':sweep,
              'features':names,'bags':a.bags,'rounds':rounds,'params':params,
              'base_pair_sha256':sha(a.baseline/'pair.txt'),
              'feature_provenance':identities,'reserve_evaluated':False,
              'model_sha256':{p.name:sha(p) for p in sorted(a.output.glob('rich_*.txt'))},
              'code_sha256':{n:sha(Path(__file__).with_name(n)) for n in ['fit_rich_context.py','rich_context.py','build_rich_context.py']}}
    atomic_json(a.output/'complete.json',complete)
    print(json.dumps(best),flush=True)


if __name__=='__main__':main()
