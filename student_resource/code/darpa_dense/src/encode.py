"""Encode supplied Parquet record tables with a frozen local checkpoint."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from train_encoder import embed, load


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8*1024*1024), b""):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--glob", default="train_*.parquet")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--max-length", type=int, default=96)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(6)
    torch.cuda.set_per_process_memory_fraction(0.65)
    tokenizer, model = load(a.model)
    model.eval()
    model_hash = sha(a.model / "model.safetensors")
    for path in sorted(a.data.glob(a.glob)):
        if not ("_s1_" in path.stem or "_targets_" in path.stem):
            continue
        done = a.output / (path.stem + ".json")
        input_hash = sha(path)
        if done.exists():
            old = json.loads(done.read_text())
            if old["model_sha256"] != model_hash or old["input_sha256"] != input_hash or old["max_length"] != a.max_length:
                raise ValueError("Encoding cache belongs to a different model/input")
            continue
        table = pq.ParquetFile(path)
        output = a.output / (path.stem + ".npy")
        partial = output.with_suffix(".npy.partial")
        count = table.metadata.num_rows
        matrix = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float16, shape=(count,model.config.hidden_size))
        offset, started = 0, time.monotonic()
        with torch.inference_mode():
            for batch in table.iter_batches(batch_size=a.batch_size,columns=["text"]):
                texts = batch.column(0).to_pylist()
                tokens = tokenizer(texts,padding=True,truncation=True,max_length=a.max_length,return_tensors="pt").to("cuda")
                vectors = embed(model,tokens).cpu().numpy()
                if not np.isfinite(vectors).all():
                    raise FloatingPointError("Nonfinite embedding")
                matrix[offset:offset+len(vectors)] = vectors
                offset += len(vectors)
                if offset % (a.batch_size*100)==0:
                    print(json.dumps({"file":path.name,"rows":offset,"total":count,"seconds":time.monotonic()-started}),flush=True)
        if offset != count:
            raise ValueError("Encoded row count differs from input")
        matrix.flush(); del matrix
        partial.replace(output)
        meta={"rows":count,"dimension":model.config.hidden_size,"dtype":"float16","model_sha256":model_hash,
              "input_sha256":input_hash,"max_length":a.max_length,"elapsed_seconds":time.monotonic()-started}
        done.write_text(json.dumps(meta,indent=2)+"\n")
        print(json.dumps({"completed":path.name,**meta}),flush=True)


if __name__ == "__main__":
    main()
