"""Learn a rival-aware owner choice and a calibrated no-match option."""
import argparse
import hashlib
import json
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from fit_owner import ROLE, fit, role_metadata
from pair_features import FEATURE_NAMES
from retrieve_owners import atomic_json


EXTRA = ["pair_probability", "pair_to_best_rival", "pair_rank",
         "pair_first", "pair_second", "pair_sum", "pair_count_05",
         "pair_entropy"]
SELECTOR_NAMES = FEATURE_NAMES + EXTRA


def selector_features(raw, probabilities, k):
    if len(raw) != len(probabilities) or len(raw) % k:
        raise ValueError("Incomplete candidate groups")
    p = np.asarray(probabilities, dtype=np.float32).reshape(-1, k)
    sorted_p = np.sort(p, axis=1)
    first, second = sorted_p[:, -1], sorted_p[:, -2]
    rival = np.where(p == first[:, None], second[:, None], first[:, None])
    rank = (p[:, None, :] > p[:, :, None]).sum(axis=2)
    total = p.sum(axis=1)
    q = p / np.maximum(total[:, None], 1e-6)
    entropy = -(q * np.log(np.maximum(q, 1e-8))).sum(axis=1)
    extras = np.stack([
        p, p-rival, rank, np.broadcast_to(first[:, None], p.shape),
        np.broadcast_to(second[:, None], p.shape),
        np.broadcast_to(total[:, None], p.shape),
        np.broadcast_to((p >= .5).sum(axis=1)[:, None], p.shape),
        np.broadcast_to(entropy[:, None], p.shape),
    ], axis=2).reshape(-1, len(EXTRA)).astype(np.float32)
    x = np.column_stack([raw, extras]).astype(np.float32)
    if not np.isfinite(x).all():
        raise FloatingPointError("Nonfinite owner selector feature")
    return x


def group_rows(table, k):
    tr = table["target_row"].to_numpy()
    ow = table["owner_row"].to_numpy()
    if len(tr) % k or (len(tr) and not np.all(tr.reshape(-1, k) == tr.reshape(-1, k)[:, :1])):
        raise ValueError("Target shortlist rows are incomplete or not contiguous")
    return tr.reshape(-1, k)[:, 0], ow.reshape(-1, k)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--features", type=Path, required=True)
    p.add_argument("--pair-model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    if (a.output/"complete.json").exists():
        raise ValueError("Selector already complete; use a new output for another experiment")
    pair = lgb.Booster(model_file=str(a.pair_model))
    db = duckdb.connect(str(a.data/"records.duckdb"), read_only=True)
    metadata, files = {}, {}
    fitting_x, fitting_y = [], []
    for source in sorted(a.data.glob("train_s1_*.parquet")):
        country = source.stem.removeprefix("train_s1_")
        folder = a.features/f"train_{country}"
        complete = json.loads((folder/"complete.json").read_text())
        config = json.loads((folder/"config.json").read_text())
        if not complete["full_competitor_shortlists"] or config["columns"] != FEATURE_NAMES:
            raise ValueError("Incomplete rival features or wrong schema")
        if "context" not in config["roles"] or "tune" not in config["roles"]:
            raise ValueError("Context fitting and tune roles are required")
        k = config["retrieval"]["k"]
        metadata[country] = (*role_metadata(db, source, a.data/f"train_targets_{country}.parquet"), k)
        files[country] = sorted(folder.glob("features-*.parquet"))
        _, roles, gold, target_roles, _ = metadata[country]
        for file in files[country]:
            table = pq.read_table(file)
            tr, ow = group_rows(table, k)
            if not len(tr):
                continue
            raw = np.column_stack([table[n].to_numpy() for n in FEATURE_NAMES]).astype(np.float32)
            pp = pair.predict(raw, num_threads=6)
            x = selector_features(raw, pp, k).reshape(-1, k, len(SELECTOR_NAMES))
            # The pair model was fitted on pair-role owners. Selector fitting
            # uses only context-role outcomes and selected encoder/unmatched
            # negatives. Tune and reserve outcomes never enter fitting.
            allowed = (target_roles[tr] == ROLE["context"]) | (
                np.isin(target_roles[tr], [-1, ROLE["encoder"]]) &
                (roles[ow[:, 0]] == ROLE["context"]))
            if allowed.any():
                fitting_x.append(x[allowed].reshape(-1, len(SELECTOR_NAMES)))
                fitting_y.append((gold[tr[allowed], None] == ow[allowed]).reshape(-1))
    x, y = np.concatenate(fitting_x), np.concatenate(fitting_y)
    selector = fit(x, y, SELECTOR_NAMES, seed=20260928, rounds=400)
    selector.save_model(str(a.output/"selector.txt"))
    atomic_json(a.output/"fit.json", {"pairs": len(y), "positive_pairs": int(y.sum()),
               "no_match_training_pairs": int((~y).sum()),
               "feature_names": SELECTOR_NAMES,
               "pair_model_sha256": hashlib.sha256(a.pair_model.read_bytes()).hexdigest(),
               "roles": ["context", "encoder/unmatched nearest-context negatives"],
               "reserve_opened": False})
    print(json.dumps({"selector_pairs": len(y), "positive_pairs": int(y.sum())}), flush=True)
    del fitting_x, fitting_y, x, y

    winners = {}
    for country, country_files in files.items():
        ids, roles, gold, target_roles, k = metadata[country]
        targets, owners, scores, margins = [], [], [], []
        for file in country_files:
            table = pq.read_table(file)
            tr, ow = group_rows(table, k)
            if not len(tr):
                continue
            raw = np.column_stack([table[n].to_numpy() for n in FEATURE_NAMES]).astype(np.float32)
            pp = pair.predict(raw, num_threads=6)
            x = selector_features(raw, pp, k)
            p = selector.predict(x, num_threads=6).reshape(-1, k)
            chosen = p.argmax(axis=1)
            ordered = np.sort(p, axis=1)
            targets.append(tr)
            owners.append(ow[np.arange(len(tr)), chosen])
            scores.append(p[np.arange(len(tr)), chosen])
            margins.append(ordered[:, -1]-ordered[:, -2])
        tr, ow, probability, margin = map(np.concatenate, (targets, owners, scores, margins))
        if len(np.unique(tr)) != len(tr):
            raise ValueError("Duplicate selected target")
        winners[country] = (tr, ow, probability)
        pq.write_table(pa.table({"target_row": tr, "owner_row": ow,
                       "selector_probability": probability, "selector_margin": margin}),
                       a.output/f"winners_{country}.parquet", compression="zstd")

    truth_count = {c: np.bincount(m[2][m[2] >= 0], minlength=len(m[0]))
                   for c, m in metadata.items()}
    sweep = []
    for threshold in np.arange(.20, .991, .01):
        total, count = 0., 0
        for country, (tr, ow, probability) in winners.items():
            ids, roles, gold, _, _ = metadata[country]
            selected = (probability >= threshold) & (roles[ow] == ROLE["tune"])
            predicted = np.bincount(ow[selected], minlength=len(ids))
            tp = np.bincount(ow[selected], weights=gold[tr[selected]] == ow[selected], minlength=len(ids))
            truth = truth_count[country]
            denominator = .25*truth + predicted
            values = np.divide(1.25*tp, denominator, out=np.ones(len(ids)), where=denominator != 0)
            mask = roles == ROLE["tune"]
            total += values[mask].sum()
            count += int(mask.sum())
        sweep.append({"threshold": round(float(threshold), 3), "macro_f05": float(total/count)})
    best = max(sweep, key=lambda row: (row["macro_f05"], row["threshold"]))
    out = a.output/"tune_matching_results.tsv"
    with out.open("w") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for country, (tr, ow, probability) in winners.items():
            ids, roles, _, _, _ = metadata[country]
            target_ids = pq.read_table(a.data/f"train_targets_{country}.parquet",
                                       columns=["entity_id"])["entity_id"].to_pylist()
            selected = (probability >= best["threshold"]) & (roles[ow] == ROLE["tune"])
            chosen = {}
            for ti, oi in zip(tr[selected], ow[selected]):
                chosen.setdefault(int(oi), []).append(target_ids[int(ti)])
            for oi in np.flatnonzero(roles == ROLE["tune"]):
                f.write(ids[oi]+"\t"+",".join(sorted(chosen.get(int(oi), [])))+"\n")
    atomic_json(a.output/"complete.json", {"best_tune": best, "threshold_sweep": sweep,
               "reserve_opened": False, "selector_chooses_among_all_k": True,
               "no_match_option": "threshold on maximal selector probability"})
    print(json.dumps(best), flush=True)


if __name__ == "__main__":
    main()
