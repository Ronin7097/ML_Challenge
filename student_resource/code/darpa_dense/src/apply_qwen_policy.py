"""Apply a frozen tune-selected Qwen owner policy to reserve or full test data."""
import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from encode import sha
from fit_owner import ROLE
from qwen_policy_tune import load_scored
from retrieve_owners import atomic_json


def compatible_config(reference, current):
    for key in ('base_weights', 'adapter_sha256', 'max_length', 'max_groups',
                'prompt_source_sha256'):
        if reference[key] != current[key]:
            raise ValueError(f'Qwen scoring config differs for {key}')
    for key in ('min', 'max', 'close_min', 'close_margin'):
        if reference['route'][key] != current['route'][key]:
            raise ValueError(f'Qwen routing changed for {key}')


def apply_choices(base, target_rows, owner_rows, scores, margins, threshold, margin,
                  reject_low, allowed):
    result = base.copy()
    for t,o,p,gap in zip(target_rows,owner_rows,scores,margins):
        target=int(t);owner=int(o)
        if p>=threshold and gap>=margin:
            if allowed[owner]:result[target]=owner
            else:result.pop(target,None)
        elif reject_low:result.pop(target,None)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--retrieval',type=Path,required=True)
    p.add_argument('--base-run',type=Path,required=True)
    p.add_argument('--qwen-scores',type=Path,required=True)
    p.add_argument('--tune-policy',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--split',choices=['test','train'],required=True)
    p.add_argument('--role',default='')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if (a.output/'run.json').exists():raise ValueError('Output already complete')
    if a.split=='train' and a.role!='reserve':
        raise ValueError('Holdout policy application is reserved for sealed reserve')
    if a.split=='test' and a.role:raise ValueError('Test role must be empty')
    frozen=json.loads(a.tune_policy.read_text())
    if (frozen.get('reserve_opened') or not frozen.get('frozen') or
            not frozen.get('selected_on_tune') or frozen.get('selected_policy')!='qwen' or
            not frozen.get('best_tune')):
        raise ValueError('Tune-selected policy manifest is not frozen')
    base_manifest=json.loads((a.base_run/'run.json').read_text())
    if base_manifest['split']!=a.split or base_manifest['role']!=a.role:
        raise ValueError('Base prediction split/role differs')
    if abs(base_manifest['threshold']-frozen['base_threshold'])>1e-12:
        raise ValueError('Frozen base threshold differs')
    if not base_manifest['reranker_routing_sha256']:
        raise ValueError('Base inference did not export frozen routing')
    country_choices={};counts={}
    for source in sorted(a.data.glob(f'{a.split}_s1_*.parquet')):
        country=source.stem.removeprefix(f'{a.split}_s1_')
        ids=pq.read_table(source,columns=['entity_id'])['entity_id'].to_pylist()
        roles=(np.asarray(pq.read_table(source,columns=['role'])['role'].to_pylist())
               if a.split=='train' else None)
        allowed=(roles==a.role if roles is not None else np.ones(len(ids),bool))
        routing=a.base_run/'routing'
        winners=routing/f'all_winners_{country}.parquet'
        decisions=routing/f'decisions_{country}.parquet'
        expected=base_manifest['reranker_routing_sha256'][country]
        if sha(winners)!=expected['all_winners'] or sha(decisions)!=expected['decisions']:
            raise ValueError('Base routing file hash changed')
        folder=a.qwen_scores/f'{a.split}_{country}'
        config=json.loads((folder/'config.json').read_text())
        reference_country=(country if country in frozen['identities']
                           else next(iter(frozen['identities'])))
        tune_reference=frozen['identities'][reference_country]['score_config']
        compatible_config(tune_reference,config)
        if config['winner_sha256']!=sha(winners):
            raise ValueError('Qwen scorer used a different winner graph')
        retrieval=json.loads((a.retrieval/f'{a.split}_{country}'/'complete.json').read_text())
        if config['retrieval']!=retrieval['signature']:
            raise ValueError('Qwen scorer used different retrieval')
        if a.split=='train' and config['route']['restrict_roles']!=['reserve']:
            raise ValueError('Reserve Qwen routing did not restrict to reserve')
        if a.split=='test' and config['route']['restrict_roles']:
            raise ValueError('Production Qwen routing restricted owner roles')
        qtr,qow,qp,qmargin=load_scored(folder)
        selection=pq.read_table(folder/'selection.parquet')
        candidates=np.load(a.retrieval/f'{a.split}_{country}'/'indices.npy',mmap_mode='r')
        if not np.array_equal(np.asarray(selection['owner_rows'].to_pylist()),candidates[qtr]):
            raise ValueError('Qwen scored owners differ from actual candidate set')
        table=pq.read_table(decisions)
        tr=table['target_row'].to_numpy();ow=table['owner_row'].to_numpy()
        probability=table['context_probability'].to_numpy()
        accepted=(probability>=base_manifest['threshold'])&allowed[ow]
        base={int(t):int(o) for t,o in zip(tr[accepted],ow[accepted])}
        if len(base)!=int(accepted.sum()):raise ValueError('Duplicate frozen decision')
        rule=frozen['best_tune']
        chosen=apply_choices(base,qtr,qow,qp,qmargin,rule['qwen_threshold'],
                             rule['qwen_margin'],rule['reject_low'],allowed)
        country_choices[country]=chosen
        counts[country]={'base_links':len(base),'final_links':len(chosen),
                         'routed_groups':len(qtr),'source1_rows':int(allowed.sum())}
    grouped={}
    for country,chosen in country_choices.items():
        source=a.data/f'{a.split}_s1_{country}.parquet'
        target=a.data/f'{a.split}_targets_{country}.parquet'
        ids=pq.read_table(source,columns=['entity_id'])['entity_id'].to_pylist()
        targets=pq.read_table(target,columns=['entity_id'])['entity_id'].to_pylist()
        for tr,ow in chosen.items():
            grouped.setdefault(ids[ow],[]).append(targets[tr])
    base_matching=a.base_run/'matching_results.tsv'
    with base_matching.open(newline='') as src,(a.output/'matching_results.tsv.partial').open('w') as dst:
        rows=csv.reader(src,delimiter='\t')
        if next(rows)!=['source1_entity_id','matched_entity_ids']:
            raise ValueError('Base matching header differs')
        dst.write('source1_entity_id\tmatched_entity_ids\n')
        n=0
        for sid,_ in rows:
            dst.write(sid+'\t'+','.join(sorted(grouped.pop(sid,[])))+'\n')
            n+=1
    if grouped or n!=base_manifest['source1_rows']:
        raise ValueError('Policy output owner coverage differs from base')
    (a.output/'matching_results.tsv.partial').replace(a.output/'matching_results.tsv')
    os.link(a.base_run/'candidate_pairs.tsv',a.output/'candidate_pairs.tsv')
    if sha(a.output/'candidate_pairs.tsv')!=base_manifest['candidate_sha256']:
        raise ValueError('Actual candidate file changed')
    atomic_json(a.output/'run.json',{'complete_submission':a.split=='test',
        'complete_holdout_predictions':a.split=='train','split':a.split,'role':a.role,
        'source1_rows':n,'countries':counts,'base_run_sha256':sha(a.base_run/'run.json'),
        'tune_policy_sha256':sha(a.tune_policy),'qwen_score_manifest_sha256':{
            country:sha(a.qwen_scores/f'{a.split}_{country}'/'complete.json') for country in counts},
        'matching_sha256':sha(a.output/'matching_results.tsv'),
        'candidate_sha256':sha(a.output/'candidate_pairs.tsv'),
        'reserve_labels_read':False})
    print(json.dumps({'source1_rows':n,'countries':counts}),flush=True)


if __name__=='__main__':main()
