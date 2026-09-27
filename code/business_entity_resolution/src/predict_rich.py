"""Apply a frozen rich route to previously scored, complete dense winners.

The candidate graph and pair winner are reused exactly. All incoming winners
contribute to context and peer evidence, including those outside the route.
"""
import argparse
import json
import time
from importlib.metadata import version
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from encode import sha
from fit_owner import CONTEXT_NAMES, FEATURE_NAMES, WINNER_NAMES, incoming_context
from retrieve_owners import atomic_json
from rich_context import EXTRA_NAMES, OMIT_BASE, enrich_row, peer_indices


def read_route_features(folder, target_rows, owners, target_count, pair=None, k=10, threads=8):
    """Read frozen winners; optionally recompute exact runner scores from rivals."""
    position=np.full(target_count,-1,np.int64);position[target_rows]=np.arange(len(target_rows))
    result=np.empty((len(target_rows),len(FEATURE_NAMES)),np.float32)
    runner=np.empty(len(target_rows),np.float32) if pair is not None else None
    seen=np.zeros(len(target_rows),np.int8)
    for path in sorted(folder.glob('features-*.parquet')):
        table=pq.read_table(path)
        tr=table['target_row'].to_numpy();ow=table['owner_row'].to_numpy()
        dest=position[tr];keep=dest>=0
        if pair is not None:
            if not np.any(keep):continue
            selected_tr=tr[keep];selected_ow=ow[keep]
            if len(selected_tr)%k or not np.all(selected_tr.reshape(-1,k)==selected_tr.reshape(-1,k)[:,:1]):
                raise ValueError('Incomplete routed competitor group')
            matrix=np.column_stack([table[n].to_numpy()[keep] for n in FEATURE_NAMES]).astype(np.float32)
            probability=pair.predict(matrix,num_threads=threads).reshape(-1,k)
            winner=probability.argmax(axis=1);group_dest=dest[keep].reshape(-1,k)[:,0]
            winner_rows=np.arange(len(winner))*k+winner
            if not np.array_equal(selected_ow[winner_rows],owners[group_dest]):
                raise ValueError('Recomputed pair winner differs from frozen cache')
            if len(np.unique(group_dest))!=len(group_dest) or np.any(seen[group_dest]):
                raise ValueError('Duplicate routed competitor group')
            result[group_dest]=matrix[winner_rows]
            runner[group_dest]=np.sort(probability,axis=1)[:,-2].astype(np.float32)
            seen[group_dest]=1
            continue
        keep[keep]=ow[keep]==owners[dest[keep]]
        dest=dest[keep]
        if len(dest):
            if len(np.unique(dest))!=len(dest) or np.any(seen[dest]):
                raise ValueError('Duplicate frozen winner feature')
            result[dest]=np.column_stack([table[n].to_numpy()[keep] for n in FEATURE_NAMES])
            seen[dest]=1
    if not np.all(seen==1):raise ValueError('Missing frozen winner features')
    return result,runner


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','retrieval','features','base-output','base-model','model','policy','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--split',choices=['train','test'],default='test')
    p.add_argument('--role',default='')
    p.add_argument('--workers',type=int,default=8)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    pa.set_cpu_count(a.workers)
    pa.set_io_thread_count(min(a.workers,8))
    if (a.output/'run.json').exists():raise ValueError('Use a new output directory')
    if (a.split=='test')==bool(a.role):raise ValueError('Train requires a role; test forbids one')
    policy=json.loads(a.policy.read_text());manifest=json.loads((a.model/'complete.json').read_text())
    if policy['rich_complete_sha256']!=sha(a.model/'complete.json'):raise ValueError('Rich model identity changed')
    models=[]
    for name,digest in manifest['model_sha256'].items():
        if sha(a.model/name)!=digest:raise ValueError('Model weights changed')
        models.append(lgb.Booster(model_file=str(a.model/name)))
    names=[n for n in CONTEXT_NAMES if n not in OMIT_BASE]+EXTRA_NAMES
    if names!=manifest['features'] or any(m.feature_name()!=names for m in models):
        raise ValueError('Model feature schema differs')
    if len(models)!=manifest['bags']:raise ValueError('Incomplete rich ensemble')
    baseline=json.loads((a.base_output/'run.json').read_text())
    if baseline['model_sha256']!=policy['base_model_sha256']:
        raise ValueError('Frozen baseline models differ')
    if sha(a.base_output/'matching_results.tsv')!=baseline['matching_sha256']:
        raise ValueError('Baseline predictions changed')
    if abs(baseline['threshold']-policy['base_threshold'])>1e-9:raise ValueError('Base threshold differs')
    if sha(a.base_model/'pair.txt')!=policy['base_pair_sha256']:raise ValueError('Pair weights changed')
    pair=lgb.Booster(model_file=str(a.base_model/'pair.txt'))
    grouped={};counts={};identities={}
    for source in sorted(a.data.glob(f'{a.split}_s1_*.parquet')):
        started=time.monotonic();c=source.stem.removeprefix(f'{a.split}_s1_')
        target=a.data/f'{a.split}_targets_{c}.parquet';retrieval=a.retrieval/f'{a.split}_{c}'
        fc=json.loads((a.features/f'{a.split}_{c}'/'config.json').read_text())
        rc=json.loads((retrieval/'complete.json').read_text())
        if fc['columns']!=FEATURE_NAMES or fc['retrieval']!=rc['signature']:
            raise ValueError('Feature and retrieval identities differ')
        if sha(source)!=rc['signature']['source_sha256'] or sha(target)!=rc['signature']['target_sha256']:
            raise ValueError('Prepared records changed')
        routing=a.base_output/'routing'
        for prefix,digest in baseline['reranker_routing_sha256'][c].items():
            if sha(routing/f'{prefix}_{c}.parquet')!=digest:raise ValueError('Base routing changed')
        w=pq.read_table(routing/f'all_winners_{c}.parquet');d=pq.read_table(routing/f'decisions_{c}.parquet')
        tr=w['target_row'].to_numpy();ow=w['owner_row'].to_numpy()
        if not np.array_equal(tr,d['target_row'].to_numpy()) or not np.array_equal(ow,d['owner_row'].to_numpy()):
            raise ValueError('Winner and decision identities differ')
        if len(np.unique(tr))!=len(tr):raise ValueError('Duplicate winner target')
        s=pq.read_table(source);t=pq.read_table(target)
        if a.split=='test' and len(tr)!=len(t):raise ValueError('Incomplete test coverage')
        wanted=np.ones(len(s),bool) if not a.role else np.asarray(s['role'].to_pylist())==a.role
        bp=d['context_probability'].to_numpy();prob=w['pair_probability'].to_numpy()
        route=(bp>policy['route_min'])&(bp<policy['route_max'])&wanted[ow]
        positions=np.flatnonzero(route);rt,ro=tr[route],ow[route]
        print(json.dumps({'country':c,'routed':len(rt),'stage':'read frozen features'}),flush=True)
        raw,runner=read_route_features(a.features/f'{a.split}_{c}',rt,ro,len(t),pair,rc['signature']['k'],a.workers)
        target_ids=t['entity_id'].to_pylist();source_ids=s['entity_id'].to_pylist()
        ss=list(zip(s['business_name'].to_pylist(),s['business_address'].to_pylist()))
        tt=list(zip(t['business_name'].to_pylist(),t['business_address'].to_pylist()))
        # Complete incoming context is computed before selecting routed rows.
        seed=np.zeros((len(tr),len(WINNER_NAMES)),np.float32)
        seed[:,WINNER_NAMES.index('pair_probability')]=prob
        seed[:,FEATURE_NAMES.index('target_source3')]=[target_ids[int(i)].startswith('S3-') for i in tr]
        context=incoming_context(ow,seed,len(s))[route,len(WINNER_NAMES):]
        del seed
        base=np.zeros((len(rt),len(CONTEXT_NAMES)),np.float32)
        base[:,:len(FEATURE_NAMES)]=raw;del raw
        base[:,WINNER_NAMES.index('pair_probability')]=prob[route]
        base[:,WINNER_NAMES.index('pair_margin')]=w['pair_margin'].to_numpy()[route]
        base[:,WINNER_NAMES.index('pair_runner_probability')]=runner
        base[:,len(WINNER_NAMES):]=context;del context
        base=base[:,[CONTEXT_NAMES.index(n) for n in CONTEXT_NAMES if n not in OMIT_BASE]]
        peers=peer_indices(ow,prob,[tt[int(i)] for i in tr])
        candidates=np.load(retrieval/'indices.npy',mmap_mode='r')
        selected=candidates[rt];member=selected==ro[:,None]
        if not np.all(member.sum(axis=1)==1):raise ValueError('Winner missing from retrieval')
        rivals=np.where(selected[:,0]!=ro,selected[:,0],selected[:,1]);del selected,member
        rich_prob=np.empty(len(rt),np.float64)
        with ProcessPoolExecutor(max_workers=a.workers) as pool:
            for start in range(0,len(rt),20000):
                end=min(start+20000,len(rt));rows=[]
                for j in range(start,end):
                    i=positions[j];peer=int(peers[i]);target_row=int(tr[i])
                    peer_row=int(tr[peer]) if peer>=0 else -1
                    rows.append((ss[int(ow[i])],tt[target_row],ss[int(rivals[j])],
                        tt[peer_row] if peer>=0 else None,float(prob[peer]) if peer>=0 else -1.,
                        float(target_ids[target_row][:2]==target_ids[peer_row][:2]) if peer>=0 else 0.))
                extra=np.asarray(list(pool.map(enrich_row,rows,chunksize=500)),np.float32)
                matrix=np.column_stack([base[start:end],extra])
                rich_prob[start:end]=np.mean([m.predict(matrix,num_threads=a.workers) for m in models],axis=0)
                if start%200000==0:print(json.dumps({'country':c,'scored':end,'total':len(rt)}),flush=True)
        if not np.isfinite(rich_prob).all():raise ValueError('Nonfinite rich probability')
        effective=(bp>=policy['base_threshold']).astype(float);effective[route]=rich_prob
        accept=(effective>=policy['best_tune']['threshold'])&wanted[ow]
        pq.write_table(pa.table({'target_row':rt,'owner_row':ro,'rich_probability':rich_prob}),
                       a.output/f'routed_{c}.parquet',compression='zstd')
        for ti,oi in zip(tr[accept],ow[accept]):grouped.setdefault(source_ids[int(oi)],[]).append(target_ids[int(ti)])
        counts[c]={'source1_rows':int(wanted.sum()),'target_rows':len(t),'routed':len(rt),
                   'matches':int(accept.sum()),'seconds':time.monotonic()-started}
        identities[c]={'source':sha(source),'target':sha(target),'retrieval':rc['signature']}
        print(json.dumps({'country':c,**counts[c]}),flush=True)
        del s,t,tt,ss,base,peers,w,d
    # Preserve original Source 1 ordering, including explicit empty results.
    rows=0;temp=a.output/'matching_results.tsv.partial'
    with (a.base_output/'matching_results.tsv').open() as src,temp.open('w') as dst:
        if next(src).rstrip('\n')!='source1_entity_id\tmatched_entity_ids':raise ValueError('Unexpected baseline header')
        dst.write('source1_entity_id\tmatched_entity_ids\n')
        for line in src:
            sid=line.split('\t',1)[0];dst.write(sid+'\t'+','.join(sorted(grouped.pop(sid,[])))+'\n');rows+=1
    if grouped or rows!=sum(v['source1_rows'] for v in counts.values()):raise ValueError('Source 1 coverage mismatch')
    temp.replace(a.output/'matching_results.tsv')
    candidate=a.base_output/'candidate_pairs.tsv'
    if sha(candidate)!=baseline['candidate_sha256']:raise ValueError('Candidate graph changed')
    (a.output/'candidate_pairs.tsv').symlink_to(candidate.resolve())
    atomic_json(a.output/'run.json',{'complete_submission':a.split=='test','split':a.split,'role':a.role,
        'source1_rows':rows,'countries':counts,'policy_sha256':sha(a.policy),
        'model_complete_sha256':sha(a.model/'complete.json'),'base_run_sha256':sha(a.base_output/'run.json'),
        'code_sha256':{n:sha(Path(__file__).with_name(n)) for n in ('predict_rich.py','rich_context.py','fit_owner.py','pair_features.py')},
        'dependency_versions':{n:version(n) for n in ('anyascii','lightgbm','numpy','pyarrow','rapidfuzz')},
        'inputs':identities,'matching_sha256':sha(a.output/'matching_results.tsv'),
        'candidate_sha256':baseline['candidate_sha256'],'reserve_labels_read':False})


if __name__=='__main__':main()
