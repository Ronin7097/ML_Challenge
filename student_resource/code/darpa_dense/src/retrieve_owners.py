"""Retrieve exact same-country owners for every target, including distractors."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from encode import sha
from train_encoder import embed, load


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--vectors", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--split", choices=["train", "test"], default="train")
    p.add_argument("--country", default="")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--max-length", type=int, default=96)
    p.add_argument("--max-rows", type=int, default=0, help="Smoke only; never marks a full run complete")
    a = p.parse_args()
    if a.k < 2 or a.batch_size < 1:
        raise ValueError("Need k>=2 and a positive batch size")
    a.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(6)
    torch.cuda.set_per_process_memory_fraction(0.65)
    tokenizer, model = load(a.model)
    model.eval()
    model_hash = sha(a.model / "model.safetensors")
    paths = sorted(a.data.glob(f"{a.split}_targets_*.parquet"))
    if not paths:
        raise ValueError("No target partitions")
    for path in paths:
        country = path.stem.removeprefix(f"{a.split}_targets_")
        if a.country and a.country != country:
            continue
        source = a.data / f"{a.split}_s1_{country}.parquet"
        encoded = a.vectors / f"{source.stem}.npy"
        vector_meta = json.loads(encoded.with_suffix(".json").read_text())
        if vector_meta["model_sha256"] != model_hash or vector_meta["input_sha256"] != sha(source) or vector_meta["max_length"] != a.max_length:
            raise ValueError("Owner vector cache differs from current model/data/settings")
        table = pq.ParquetFile(path)
        n = table.metadata.num_rows
        dest = a.output / f"{a.split}_{country}"
        dest.mkdir(exist_ok=True)
        signature = {"model_sha256": model_hash, "target_sha256": sha(path),
                     "source_sha256": vector_meta["input_sha256"], "rows": n,
                     "owner_rows": vector_meta["rows"], "k": a.k,
                     "max_length": a.max_length, "batch_size": a.batch_size,
                     "retrieval": "exhaustive normalized float16 GPU inner product"}
        completion = dest / "complete.json"
        if completion.exists():
            if json.loads(completion.read_text())["signature"] != signature:
                raise ValueError("Completed retrieval uses different inputs/settings")
            continue
        progress_path = dest / "progress.json"
        offset = 0
        if progress_path.exists():
            progress = json.loads(progress_path.read_text())
            if progress["signature"] != signature:
                raise ValueError("Partial retrieval uses different inputs/settings")
            offset = progress["rows_done"]
        mode = "r+" if offset else "w+"
        indices = np.lib.format.open_memmap(dest / "indices.npy", mode=mode, dtype=np.int32, shape=(n, a.k))
        values = np.lib.format.open_memmap(dest / "scores.npy", mode=mode, dtype=np.float32, shape=(n, a.k))
        raw = np.load(encoded, mmap_mode="r")
        if len(raw) < a.k:
            raise ValueError("Country has fewer owners than k")
        reference = F.normalize(torch.tensor(np.asarray(raw), device="cuda", dtype=torch.float16), dim=1)
        started = time.monotonic()
        base = 0
        with torch.inference_mode():
            for batch in table.iter_batches(batch_size=a.batch_size, columns=["text"]):
                end = base + len(batch)
                if end <= offset:
                    base = end
                    continue
                if base != offset:
                    raise ValueError("Resume offset must be a batch boundary")
                tokens = tokenizer(batch.column(0).to_pylist(), padding=True, truncation=True,
                                   max_length=a.max_length, return_tensors="pt").to("cuda")
                query = F.normalize(embed(model, tokens).to(torch.float16), dim=1)
                scores, owners = torch.topk(query @ reference.T, k=a.k, dim=1, sorted=True)
                if not torch.isfinite(scores).all():
                    raise FloatingPointError("Nonfinite retrieval score")
                indices[base:end] = owners.cpu().numpy()
                values[base:end] = scores.float().cpu().numpy()
                offset = end
                base = end
                if end % (a.batch_size * 20) == 0 or end == n or (a.max_rows and end >= a.max_rows):
                    indices.flush(); values.flush()
                    progress = {"signature": signature, "rows_done": end,
                                "seconds_this_session": time.monotonic()-started}
                    atomic_json(progress_path, progress)
                    print(json.dumps({"country": country, "rows": end, "total": n,
                                      "seconds": progress["seconds_this_session"]}), flush=True)
                if a.max_rows and end >= a.max_rows:
                    break
        del reference
        torch.cuda.empty_cache()
        indices.flush(); values.flush()
        if offset == n:
            atomic_json(completion, {"signature": signature, "rows_done": n,
                                     "full_target_pool": True, "full_owner_pool": True})
        elif not a.max_rows:
            raise RuntimeError("Retrieval ended before every target was processed")


if __name__ == "__main__":
    main()
