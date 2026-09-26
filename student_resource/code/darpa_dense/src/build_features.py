"""Score-ready top-k features; include every rival of every selected target."""
import argparse
import hashlib
import json
import multiprocessing as mp
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from encode import sha
from pair_features import FEATURE_NAMES, normalized, shortlist
from retrieve_owners import atomic_json

SOURCE_NAMES = SOURCE_ADDRESSES = SOURCE_FREQ = None


def work(items):
    result = []
    for name, address, source3, indices, scores in items:
        records = [(SOURCE_NAMES[i], SOURCE_ADDRESSES[i]) for i in indices]
        result.append(shortlist(records, (name, address), scores, SOURCE_FREQ[indices], source3))
    return np.concatenate(result, axis=0)


def main():
    global SOURCE_NAMES, SOURCE_ADDRESSES, SOURCE_FREQ
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--retrieval", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--split", choices=["train", "test"], default="train")
    p.add_argument("--roles", nargs="+", default=["pair", "context", "tune"])
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--chunk-size", type=int, default=65536)
    p.add_argument("--max-chunks", type=int, default=0, help="Incomplete smoke run")
    a = p.parse_args()
    if "reserve" in a.roles:
        raise ValueError("Reserve feature export requires a separately frozen evaluation workflow")
    a.output.mkdir(parents=True, exist_ok=True)
    code_hash = hashlib.sha256(Path(__file__).with_name("pair_features.py").read_bytes()).hexdigest()
    for source in sorted(a.data.glob(f"{a.split}_s1_*.parquet")):
        country = source.stem.removeprefix(f"{a.split}_s1_")
        retrieval = a.retrieval/f"{a.split}_{country}"
        complete = json.loads((retrieval/"complete.json").read_text())
        if not complete["full_target_pool"] or not complete["full_owner_pool"]:
            raise ValueError("Features require complete full-pool retrieval")
        target = a.data/f"{a.split}_targets_{country}.parquet"
        sig = complete["signature"]
        if sig["source_sha256"] != sha(source) or sig["target_sha256"] != sha(target):
            raise ValueError("Retrieval row identities differ from current input")
        dest = a.output/f"{a.split}_{country}"
        dest.mkdir(exist_ok=True)
        signature = {"retrieval": sig, "feature_code_sha256": code_hash, "columns": FEATURE_NAMES,
                     "roles": a.roles if a.split == "train" else ["all"], "chunk_size": a.chunk_size}
        config = dest/"config.json"
        if config.exists() and json.loads(config.read_text()) != signature:
            raise ValueError("Feature output directory belongs to another configuration")
        atomic_json(config, signature)
        if (dest/"complete.json").exists():
            continue
        s = pq.read_table(source).to_pydict()
        SOURCE_NAMES, SOURCE_ADDRESSES = s["business_name"], s["business_address"]
        normalized_names = [normalized(n) for n in SOURCE_NAMES]
        counts = Counter(normalized_names)
        SOURCE_FREQ = np.asarray([counts[n] for n in normalized_names], dtype=np.int32)
        del counts, normalized_names
        selected = np.isin(np.asarray(s["role"]), a.roles) if a.split == "train" else np.ones(len(SOURCE_NAMES), bool)
        indices = np.load(retrieval/"indices.npy", mmap_mode="r")
        scores = np.load(retrieval/"scores.npy", mmap_mode="r")
        k = sig["k"]
        total_pairs, total_targets, offset, chunks = 0, 0, 0, 0
        started = time.monotonic()
        # Fork shares the read-only reference strings; workers cache normalization.
        with mp.get_context("fork").Pool(a.workers) as pool:
            for chunk, batch in enumerate(pq.ParquetFile(target).iter_batches(batch_size=a.chunk_size,
                                                                            columns=["entity_id", "business_name", "business_address"])):
                path = dest/f"features-{chunk:05d}.parquet"
                if path.exists():
                    n = pq.ParquetFile(path).metadata.num_rows
                    total_pairs += n; total_targets += n//k
                    offset += len(batch); chunks += 1
                    continue
                stop = offset+len(batch)
                keep = np.flatnonzero(selected[indices[offset:stop]].any(axis=1))
                cols = batch.to_pydict()
                row_indices = keep+offset
                items = [(cols["business_name"][j], cols["business_address"][j],
                          cols["entity_id"][j].startswith("S3-"), indices[offset+j], scores[offset+j]) for j in keep]
                groups = [items[start:start+128] for start in range(0, len(items), 128)]
                matrices = list(pool.imap(work, groups, chunksize=1))
                matrix = np.concatenate(matrices) if matrices else np.empty((0, len(FEATURE_NAMES)), np.float32)
                arrays = {"target_row": pa.array(np.repeat(row_indices, k)),
                          "owner_row": pa.array(np.asarray(indices[row_indices]).reshape(-1)),
                          **{name: pa.array(matrix[:,j]) for j, name in enumerate(FEATURE_NAMES)}}
                temp = path.with_suffix(".parquet.partial")
                pq.write_table(pa.table(arrays), temp, compression="zstd")
                temp.replace(path)
                total_pairs += len(matrix); total_targets += len(keep)
                offset = stop; chunks += 1
                progress = {"country": country, "scanned_targets": offset, "selected_targets": total_targets,
                            "candidate_pairs": total_pairs, "seconds": time.monotonic()-started}
                atomic_json(dest/"progress.json", progress)
                print(json.dumps(progress), flush=True)
                if a.max_chunks and chunks >= a.max_chunks:
                    break
        if offset == sig["rows"]:
            atomic_json(dest/"complete.json", {"rows_scanned": offset, "selected_targets": total_targets,
                                               "candidate_pairs": total_pairs, "full_competitor_shortlists": True,
                                               "reserve_labels_opened": False})
        elif not a.max_chunks:
            raise ValueError("Not all targets scanned")


if __name__ == "__main__":
    main()
