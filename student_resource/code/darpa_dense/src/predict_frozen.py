"""Apply frozen owner/context models to full test or sealed reserve candidates."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pyarrow.parquet as pq

from fit_owner import CONTEXT_NAMES, FEATURE_NAMES, WINNER_NAMES, incoming_context
from retrieve_owners import atomic_json


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()


def score_country(source,target,folder,pair,contexts,k):
    ids=pq.read_table(source,columns=['entity_id'])['entity_id'].to_pylist()
    targets=pq.read_table(target,columns=['entity_id'])['entity_id'].to_pylist()
    tr_chunks=[];ow_chunks=[];x_chunks=[]
    for path in sorted(folder.glob('features-*.parquet')):
        table=pq.read_table(path)
        tr=table['target_row'].to_numpy();ow=table['owner_row'].to_numpy()
        if len(tr)%k or (len(tr) and not np.all(tr.reshape(-1,k)==tr.reshape(-1,k)[:,:1])):
            raise ValueError(f'Incomplete rival group in {path}')
        if not len(tr):
            continue
        raw=np.column_stack([table[n].to_numpy() for n in FEATURE_NAMES]).astype(np.float32)
        p=pair.predict(raw,num_threads=6).reshape(-1,k)
        winner=p.argmax(axis=1)
        selected=np.arange(len(winner))*k+winner
        sorted_p=np.sort(p,axis=1)
        first,second=sorted_p[:,-1],sorted_p[:,-2]
        extra=np.column_stack([first,second,first-second,p.sum(axis=1),
                               (p>=.5).sum(axis=1),(p>=.1).sum(axis=1)])
        tr_chunks.append(tr[selected]);ow_chunks.append(ow[selected])
        x_chunks.append(np.column_stack([raw[selected],extra]).astype(np.float32))
    tr,ow,x=map(np.concatenate,(tr_chunks,ow_chunks,x_chunks))
    if x.shape[1]!=len(WINNER_NAMES) or len(np.unique(tr))!=len(tr):
        raise ValueError('Winner feature schema or target uniqueness failed')
    x=incoming_context(ow,x,len(ids))
    if x.shape[1]!=len(CONTEXT_NAMES):
        raise ValueError('Context feature schema changed')
    prob=np.mean([m.predict(x,num_threads=6) for m in contexts],axis=0)
    if not np.isfinite(prob).all():
        raise FloatingPointError('Nonfinite context probability')
    return ids,targets,tr,ow,prob


def candidate_map(ids,targets,retrieval,selected_owners):
    complete=json.loads((retrieval/'complete.json').read_text())
    if not complete['full_target_pool'] or not complete['full_owner_pool']:
        raise ValueError('Incomplete candidate set')
    k=complete['signature']['k']
    indices=np.load(retrieval/'indices.npy',mmap_mode='r')
    if indices.shape!=(len(targets),k) or complete['signature']['owner_rows']!=len(ids):
        raise ValueError('Candidate row identity mismatch')
    flat=indices.reshape(-1)
    if np.any(flat<0) or np.any(flat>=len(ids)):
        raise ValueError('Candidate owner row outside source')
    order=np.argsort(flat,kind='stable')
    sorted_owner=flat[order]
    result={}
    wanted=np.flatnonzero(selected_owners)
    starts=np.searchsorted(sorted_owner,wanted,side='left')
    ends=np.searchsorted(sorted_owner,wanted,side='right')
    for owner,begin,end in zip(wanted,starts,ends):
        target_rows=order[begin:end]//k
        result[ids[owner]]=','.join(targets[i] for i in target_rows)
    return result,int((ends-starts).sum())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--features',type=Path,required=True)
    p.add_argument('--retrieval',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--source1-tsv',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--split',choices=['test','train'],default='test')
    p.add_argument('--role',default='',help='For sealed train evaluation, e.g. reserve')
    p.add_argument('--threshold',type=float,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/'run.json').exists():
        raise ValueError('Output run already recorded; use a new output directory')
    if a.split=='test' and a.role:
        raise ValueError('Test must cover every Source 1 entity')
    if a.split=='train' and not a.role:
        raise ValueError('Train evaluation needs an explicit output role')
    frozen=json.loads((a.model/'complete.json').read_text())
    if abs(frozen['best_tune']['threshold']-a.threshold)>1e-12:
        raise ValueError('Threshold differs from frozen tune decision')
    pair=lgb.Booster(model_file=str(a.model/'pair.txt'))
    contexts=[lgb.Booster(model_file=str(path)) for path in sorted(a.model.glob('context_*.txt'))]
    if len(contexts)!=frozen['bags']:
        raise ValueError('Context ensemble is incomplete')
    matching={};candidates={};counts={}
    for source in sorted(a.data.glob(f'{a.split}_s1_*.parquet')):
        country=source.stem.removeprefix(f'{a.split}_s1_')
        target=a.data/f'{a.split}_targets_{country}.parquet'
        folder=a.features/f'{a.split}_{country}'
        retrieval=a.retrieval/f'{a.split}_{country}'
        feature_complete=json.loads((folder/'complete.json').read_text())
        feature_config=json.loads((folder/'config.json').read_text())
        retrieval_complete=json.loads((retrieval/'complete.json').read_text())
        if not feature_complete['full_competitor_shortlists'] or feature_config['columns']!=FEATURE_NAMES:
            raise ValueError('Incomplete rival feature graph')
        if feature_config['retrieval']!=retrieval_complete['signature']:
            raise ValueError('Features and retrieval differ')
        if feature_config['retrieval']['source_sha256']!=sha(source) or feature_config['retrieval']['target_sha256']!=sha(target):
            raise ValueError('Prepared input changed')
        if a.split=='test' and feature_complete['rows_scanned']!=feature_complete['selected_targets']:
            raise ValueError('Not every test target was scored')
        if a.split=='train' and a.role not in feature_config['roles']:
            raise ValueError('Requested holdout role is absent from feature graph')
        k=feature_config['retrieval']['k']
        ids,targets,tr,ow,prob=score_country(source,target,folder,pair,contexts,k)
        if a.split=='train':
            roles=np.asarray(pq.read_table(source,columns=['role'])['role'].to_pylist())
            wanted=roles==a.role
        else:wanted=np.ones(len(ids),bool)
        accepted=(prob>=a.threshold)&wanted[ow]
        grouped={}
        for ti,oi in zip(tr[accepted],ow[accepted]):
            grouped.setdefault(ids[int(oi)],[]).append(targets[int(ti)])
        for key,value in grouped.items():
            if key in matching:
                raise ValueError('Source 1 ID occurs in two countries')
            matching[key]=','.join(sorted(value))
        candidate_country,n_pairs=candidate_map(ids,targets,retrieval,wanted)
        if any(key in candidates for key in candidate_country):
            raise ValueError('Duplicate Source 1 ID in candidate groups')
        candidates.update(candidate_country)
        counts[country]={'source1_rows':int(wanted.sum()),'target_rows':len(targets),
                         'candidate_pairs':n_pairs,'matches':int(accepted.sum())}
        print(json.dumps({'country':country,**counts[country]}),flush=True)
    match_temp=a.output/'matching_results.tsv.partial'
    cand_temp=a.output/'candidate_pairs.tsv.partial'
    n=0
    with a.source1_tsv.open(newline='') as src,match_temp.open('w') as mat,cand_temp.open('w') as can:
        reader=csv.reader(src,delimiter='\t')
        if next(reader)!=['entity_id','business_name','business_address','country']:
            raise ValueError('Unexpected original Source 1 header')
        mat.write('source1_entity_id\tmatched_entity_ids\n')
        can.write('source1_entity_id\tcandidate_entity_ids\n')
        for row in reader:
            if len(row)!=4:
                raise ValueError('Malformed original Source 1 row')
            sid=row[0]
            if sid not in candidates:
                if a.split=='train':continue
                raise ValueError(f'Missing test Source 1 ID {sid}')
            mat.write(sid+'\t'+matching.get(sid,'')+'\n')
            can.write(sid+'\t'+candidates.pop(sid)+'\n')
            n+=1
    if candidates or n!=sum(x['source1_rows'] for x in counts.values()):
        raise ValueError('Output Source 1 coverage failed')
    match_temp.replace(a.output/'matching_results.tsv')
    cand_temp.replace(a.output/'candidate_pairs.tsv')
    atomic_json(a.output/'run.json',{'complete_submission':a.split=='test',
        'complete_holdout_predictions':a.split=='train','split':a.split,'role':a.role,
        'threshold':a.threshold,'source1_rows':n,'countries':counts,
        'model_sha256':{'pair.txt':sha(a.model/'pair.txt'),
                         **{path.name:sha(path) for path in sorted(a.model.glob('context_*.txt'))}},
        'matching_sha256':sha(a.output/'matching_results.tsv'),
        'candidate_sha256':sha(a.output/'candidate_pairs.tsv'),
        'reserve_labels_read':False})


if __name__=='__main__':main()
