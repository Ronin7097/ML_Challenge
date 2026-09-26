"""Fit pair and incoming-link decisions on distinct roles; tune without reserve."""
import argparse
import json
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from pair_features import FEATURE_NAMES
from prepare import sql
from retrieve_owners import atomic_json

ROLE = {"encoder": 0, "pair": 1, "context": 2, "tune": 3, "reserve": 4}
WINNER_NAMES = FEATURE_NAMES + ["pair_probability", "pair_runner_probability", "pair_margin",
                                "pair_sum", "pair_count_05", "pair_count_01"]
CONTEXT_EXTRA = ["other_incoming_count", "other_probability_sum", "best_other_probability",
                 "other_count_05", "other_count_09", "other_same_source_count_05",
                 "other_opposite_source_count_05", "incoming_rank"]
CONTEXT_NAMES = WINNER_NAMES + CONTEXT_EXTRA


def role_metadata(db, source, target):
    # Split membership is used for exclusion. Reserve outcomes are never scored.
    db.execute(f"CREATE OR REPLACE TEMP VIEW s AS SELECT row_number() OVER(ORDER BY entity_id)-1 AS owner_row,* FROM read_parquet({sql(source)})")
    db.execute(f"CREATE OR REPLACE TEMP VIEW t AS SELECT row_number() OVER(ORDER BY entity_id)-1 AS target_row,* FROM read_parquet({sql(target)})")
    s = db.execute("SELECT entity_id,role FROM s ORDER BY owner_row").fetchall()
    r = db.execute("""SELECT coalesce(s.owner_row,-1),coalesce(s.role,'unmatched') FROM t
                      LEFT JOIN owners o ON t.entity_id=o.target_id
                      LEFT JOIN s ON o.s1_id=s.entity_id ORDER BY t.target_row""").fetchall()
    owner_roles = np.asarray([ROLE[x[1]] for x in s], np.int8)
    gold_owner = np.asarray([x[0] for x in r], np.int32)
    target_roles = np.asarray([ROLE.get(x[1],-1) for x in r], np.int8)
    return [x[0] for x in s], owner_roles, gold_owner, target_roles


def fit(x, y, names, seed, rounds=400):
    if len(np.unique(y)) != 2:
        raise ValueError("Fitting data must include positive and negative pairs")
    return lgb.train({"objective":"binary", "metric":"binary_logloss", "verbosity":-1,
                      "num_threads":6, "learning_rate":0.05, "num_leaves":31,
                      "min_data_in_leaf":100, "lambda_l2":5, "feature_fraction":0.8,
                      "bagging_fraction":0.8, "bagging_freq":1, "seed":seed,
                      "deterministic":True, "force_col_wise":True},
                     lgb.Dataset(x, label=y, feature_name=names), num_boost_round=rounds)


def incoming_context(owner, x, total_owners):
    """All winners of selected owners are present; each edge excludes itself."""
    p = x[:, WINNER_NAMES.index("pair_probability")]
    s3 = x[:, FEATURE_NAMES.index("target_source3")].astype(bool)
    count = np.bincount(owner, minlength=total_owners)
    prob_sum = np.bincount(owner, weights=p, minlength=total_owners)
    c05 = np.bincount(owner, weights=p>=.5, minlength=total_owners)
    c09 = np.bincount(owner, weights=p>=.9, minlength=total_owners)
    s305 = np.bincount(owner, weights=(p>=.5)&s3, minlength=total_owners)
    s205 = c05-s305
    maximum = np.full(total_owners, -1.0)
    np.maximum.at(maximum, owner, p)
    top = p == maximum[owner]
    maxcount = np.bincount(owner, weights=top, minlength=total_owners)
    second = np.full(total_owners, -1.0)
    np.maximum.at(second, owner[~top], p[~top])
    othermax = np.where(top & (maxcount[owner]==1), second[owner], maximum[owner])
    order = np.lexsort((-p, owner))
    ordered_owner = owner[order]
    starts = np.maximum.accumulate(np.where(np.r_[True, ordered_owner[1:]!=ordered_owner[:-1]], np.arange(len(owner)), 0))
    ties = np.maximum.accumulate(np.where(np.r_[True,
        (ordered_owner[1:]!=ordered_owner[:-1]) | (p[order][1:]!=p[order][:-1])], np.arange(len(owner)), 0))
    rank = np.empty(len(owner), np.float32)
    rank[order] = ties-starts  # Equal probabilities must not encode target-ID order.
    same = np.where(s3, s305[owner], s205[owner])-(p>=.5)
    opposite = np.where(s3, s205[owner], s305[owner])
    extra = np.column_stack([np.log1p(count[owner]-1), prob_sum[owner]-p, othermax,
                             c05[owner]-(p>=.5), c09[owner]-(p>=.9), same, opposite, rank])
    return np.column_stack([x,extra]).astype(np.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--bags", type=int, default=5)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    if (a.output/"complete.json").exists():
        raise ValueError("Fitting already complete; use a new output for another experiment")
    db = duckdb.connect(str(a.data/"records.duckdb"),read_only=True)
    metadata, feature_files = {}, {}
    fit_x, fit_y = [], []
    for source in sorted(a.data.glob("train_s1_*.parquet")):
        country = source.stem.removeprefix("train_s1_")
        target = a.data/f"train_targets_{country}.parquet"
        folder = a.features/f"train_{country}"
        completed = json.loads((folder/"complete.json").read_text())
        config = json.loads((folder/"config.json").read_text())
        if not completed["full_competitor_shortlists"] or config["columns"] != FEATURE_NAMES:
            raise ValueError("Incomplete feature graph or incompatible schema")
        if not set(("pair","context","tune")).issubset(config["roles"]):
            raise ValueError("Feature graph omits fitting or tune roles")
        metadata[country] = role_metadata(db, source, target)
        ids, roles, gold, gold_roles = metadata[country]
        feature_files[country] = sorted(folder.glob("features-*.parquet"))
        for file in feature_files[country]:
            table = pq.read_table(file)
            tr = table["target_row"].to_numpy(); ow = table["owner_row"].to_numpy()
            # Positive fitting targets and realistic hard negatives whose nearest
            # owner is in pair-fit. No held-out target is used for fitting.
            k = config["retrieval"]["k"]
            if len(tr)%k or (len(tr) and not np.all(tr.reshape(-1,k)==tr.reshape(-1,k)[:,:1])):
                raise ValueError("Target shortlist rows are not complete and contiguous")
            nearest = np.repeat(roles[ow.reshape(-1,k)[:,0]]==ROLE["pair"], k)
            allowed = (gold_roles[tr]==ROLE["pair"]) | (((gold_roles[tr]==-1)|(gold_roles[tr]==ROLE["encoder"])) & nearest)
            if allowed.any():
                fit_x.append(np.column_stack([table[n].to_numpy()[allowed] for n in FEATURE_NAMES]))
                fit_y.append(gold[tr[allowed]]==ow[allowed])
    x,y = np.concatenate(fit_x),np.concatenate(fit_y)
    pair = fit(x,y,FEATURE_NAMES,20260927)
    pair.save_model(str(a.output/"pair.txt"))
    pair_report = {"rows":len(y),"positive_pairs":int(y.sum()),"features":FEATURE_NAMES}
    atomic_json(a.output/"pair_fit.json",pair_report)
    print(json.dumps({"pair_fit":len(y),"positive_pairs":int(y.sum())}),flush=True)
    del fit_x, fit_y, x, y
    winners = {}
    ctx_x,ctx_y = [],[]
    for country, files in feature_files.items():
        ids,roles,gold,gold_roles = metadata[country]
        chunks=[];target_rows=[];owner_rows=[]
        config=json.loads((a.features/f"train_{country}"/"config.json").read_text());k=config["retrieval"]["k"]
        for file in files:
            table=pq.read_table(file)
            if not len(table):continue
            x=np.column_stack([table[n].to_numpy() for n in FEATURE_NAMES])
            probabilities=pair.predict(x,num_threads=6).reshape(-1,k)
            winner=probabilities.argmax(axis=1)
            offset=np.arange(len(winner))*k+winner
            sorted_p=np.sort(probabilities,axis=1)
            p1,p2=sorted_p[:,-1],sorted_p[:,-2]
            extra=np.column_stack([p1,p2,p1-p2,probabilities.sum(axis=1),(probabilities>=.5).sum(axis=1),(probabilities>=.1).sum(axis=1)])
            chunks.append(np.column_stack([x[offset],extra]))
            target_rows.append(table["target_row"].to_numpy()[offset]);owner_rows.append(table["owner_row"].to_numpy()[offset])
        tr,ow=np.concatenate(target_rows),np.concatenate(owner_rows)
        x=incoming_context(ow,np.concatenate(chunks),len(ids))
        # Context of selected owners is complete: any target capable of selecting
        # them was included with all k rivals. Other owners' aggregates are partial
        # and MUST NOT be fitted or evaluated.
        selected=np.isin(roles[ow],[ROLE["pair"],ROLE["context"],ROLE["tune"]])
        tr,ow,x=tr[selected],ow[selected],x[selected]
        winners[country]=(tr,ow,x)
        pq.write_table(pa.table({"target_row":tr,"owner_row":ow,**{n:x[:,j] for j,n in enumerate(CONTEXT_NAMES)}}),
                       a.output/f"winners_{country}.parquet",compression="zstd")
        mask=(roles[ow]==ROLE["context"]) & np.isin(gold_roles[tr],[-1,ROLE["encoder"],ROLE["context"]])
        ctx_x.append(x[mask]);ctx_y.append(gold[tr[mask]]==ow[mask])
    x,y=np.concatenate(ctx_x),np.concatenate(ctx_y)
    context=[]
    for seed in range(a.bags):
        model=fit(x,y,CONTEXT_NAMES,20260927+seed)
        model.save_model(str(a.output/f"context_{seed}.txt"));context.append(model)
    atomic_json(a.output/"context_fit.json",{"rows":len(y),"positive_pairs":int(y.sum()),"features":CONTEXT_NAMES,"bags":a.bags})
    tune=[]
    for country,(tr,ow,x) in winners.items():
        ids,roles,gold,gold_roles=metadata[country]
        probability=np.mean([m.predict(x,num_threads=6) for m in context],axis=0)
        pq.write_table(pa.table({"target_row":tr,"owner_row":ow,"context_probability":probability}),a.output/f"decisions_{country}.parquet",compression="zstd")
        mask=roles[ow]==ROLE["tune"]
        tune.append((country,tr[mask],ow[mask],probability[mask]))
    gold_counts={c:np.bincount(m[2][m[2]>=0],minlength=len(m[0])) for c,m in metadata.items()}
    sweep=[]
    for threshold in np.arange(.30,.991,.01):
        total,count=0.,0
        for country,tr,ow,pred_p in tune:
            ids,roles,gold,_=metadata[country]
            accept=pred_p>=threshold
            pred=np.bincount(ow[accept],minlength=len(ids))
            tp=np.bincount(ow[accept],weights=gold[tr[accept]]==ow[accept],minlength=len(ids))
            truth=gold_counts[country];den=.25*truth+pred
            values=np.divide(1.25*tp,den,out=np.ones(len(ids)),where=den!=0)
            selected=roles==ROLE["tune"]
            total+=values[selected].sum();count+=int(selected.sum())
        sweep.append({"threshold":round(float(threshold),3),"macro_f05":float(total/count)})
    best=max(sweep,key=lambda r:(r["macro_f05"],r["threshold"]))
    output=a.output/"tune_matching_results.tsv"
    with output.open("w") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for country,tr,ow,probability in tune:
            ids,roles,_,_=metadata[country]
            tids=pq.read_table(a.data/f"train_targets_{country}.parquet",columns=["entity_id"])["entity_id"].to_pylist()
            chosen={}
            for ti,oi in zip(tr[probability>=best["threshold"]],ow[probability>=best["threshold"]]):
                chosen.setdefault(int(oi),[]).append(tids[int(ti)])
            for oi in np.flatnonzero(roles==ROLE["tune"]):
                f.write(ids[oi]+"\t"+",".join(sorted(chosen.get(int(oi),[])))+"\n")
    atomic_json(a.output/"complete.json",{"stage":"dense owner plus incoming context; Qwen pending",
               "best_tune":best,"threshold_sweep":sweep,"reserve_evaluated":False,
               "pair_rows":pair_report["rows"],"context_rows":len(y),"bags":a.bags})
    print(json.dumps(best),flush=True)


if __name__=="__main__":
    main()
