"""Train a target-to-owner multilingual encoder only on the fitting role."""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

MODEL = "ibm-granite/granite-embedding-97m-multilingual-r2"
REVISION = "835ad14087e140460703cf0fae09f97d469d65c2"


def embed(model, tokens):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        cls = model(**tokens).last_hidden_state[:, 0]
    return F.normalize(cls.float(), dim=-1)


def gradient_cache(model, batches, chunk_size, duplicate_mask, temperature=0.05):
    """Replay dropout RNG so microbatches yield the full-batch contrastive gradient."""
    encoded, replay = [], []
    for tokens in batches:
        pieces, states = [], []
        for start in range(0, len(tokens["input_ids"]), chunk_size):
            block = {k: v[start:start+chunk_size] for k, v in tokens.items()}
            states.append((block, torch.get_rng_state(), torch.cuda.get_rng_state()))
            with torch.no_grad():
                pieces.append(embed(model, block))
        encoded.append(torch.cat(pieces).detach().requires_grad_())
        replay.append(states)
    scores = encoded[0] @ encoded[1].T / temperature
    scores = scores.masked_fill(duplicate_mask, float("-inf"))
    loss = F.cross_entropy(scores, torch.arange(len(scores), device=scores.device))
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite contrastive loss")
    loss.backward()
    for representation, states in zip(encoded, replay):
        for index, (block, cpu_rng, cuda_rng) in enumerate(states):
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.set_rng_state(cpu_rng)
                torch.cuda.set_rng_state(cuda_rng)
                current = embed(model, block)
                gradient = representation.grad[index*chunk_size:index*chunk_size+len(current)]
                (current * gradient).sum().backward()
    return float(loss.detach())


def load(path):
    config = AutoConfig.from_pretrained(path, local_files_only=True)
    config.reference_compile = False
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModel.from_pretrained(path, config=config, local_files_only=True, attn_implementation="sdpa").cuda()
    return tokenizer, model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pairs", type=Path, required=True)
    p.add_argument("--base-model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--microbatch", type=int, default=64)
    p.add_argument("--max-length", type=int, default=96)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=7e-5)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    if (a.output / "complete.json").exists():
        raise SystemExit("Training already complete")
    if (a.output / "progress.json").exists() and not a.resume:
        raise SystemExit("Existing training run: use --resume or a fresh output")
    torch.set_num_threads(6)
    torch.set_float32_matmul_precision("high")
    torch.cuda.set_per_process_memory_fraction(0.65)
    torch.manual_seed(20260927)
    np.random.seed(20260927)
    rows = pq.read_table(a.pairs).to_pydict()
    if len(set(rows["s1_id"])) != len(rows["s1_id"]):
        raise ValueError("Encoder input repeats an owner")
    countries = np.asarray(rows["country"])
    groups = {c: np.flatnonzero(countries == c) for c in sorted(set(countries))}
    rng = np.random.default_rng(20260927)
    schedule = []
    for _ in range(a.epochs):
        orders = {c: rng.permutation(ids) for c, ids in groups.items()}
        offsets = {c: 0 for c in groups}
        for _ in range(max(math.ceil(len(v)/a.batch_size) for v in groups.values())):
            for c in groups:
                if offsets[c] >= len(orders[c]):
                    orders[c] = rng.permutation(groups[c]); offsets[c] = 0
                idx = orders[c][offsets[c]:offsets[c]+a.batch_size]
                offsets[c] += len(idx)
                if len(idx) > 1:
                    schedule.append((c, idx))
    if a.max_steps:
        schedule = schedule[:a.max_steps]
    checkpoint = a.output / "checkpoint"
    tokenizer, model = load(checkpoint if a.resume and (checkpoint / "training_state.pt").exists() else a.base_model)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.learning_rate, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, max(1,len(schedule)//10), len(schedule))
    first_step = 0
    if a.resume and (checkpoint / "training_state.pt").exists():
        state = torch.load(checkpoint / "training_state.pt", weights_only=False, map_location="cpu")
        optimizer.load_state_dict(state["optimizer"]); scheduler.load_state_dict(state["scheduler"])
        first_step = state["step"]
        torch.set_rng_state(state["cpu_rng"]); torch.cuda.set_rng_state(state["cuda_rng"])
    config = {"model": MODEL, "revision": REVISION, "parameters":sum(p.numel() for p in model.parameters()),
              "seed":20260927, "pairs":len(countries), "steps":len(schedule),
              "batch_size":a.batch_size,"microbatch":a.microbatch,"max_length":a.max_length,
              "learning_rate":a.learning_rate,"epochs":a.epochs,"max_steps":a.max_steps,
              "country_counts":{c:len(v) for c,v in groups.items()},"pooling":"CLS, L2 normalized",
              "loss":"target-to-owner in-batch cross entropy, temperature 0.05, exact gradient caching",
              "input_sha256":hashlib.sha256(a.pairs.read_bytes()).hexdigest()}
    (a.output / "training_config.json").write_text(json.dumps(config,indent=2)+"\n")
    started = time.monotonic()

    def save(step, final=False):
        target = a.output / ("model" if final else "checkpoint")
        model.save_pretrained(target); tokenizer.save_pretrained(target)
        if not final:
            torch.save({"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),"step":step,
                        "cpu_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state()},target/"training_state.pt")

    for step, (country, indices) in enumerate(schedule):
        if step < first_step:
            continue
        texts = [[rows[key][int(i)] for i in indices] for key in ("target_text", "source_text")]
        tokens = [tokenizer(t,padding=True,truncation=True,max_length=a.max_length,return_tensors="pt").to("cuda") for t in texts]
        hashes = [np.asarray([int.from_bytes(hashlib.blake2b(t.encode(),digest_size=8).digest(),"little") for t in side],dtype=np.uint64) for side in texts]
        duplicate = (hashes[0][:,None]==hashes[0][None,:]) | (hashes[1][:,None]==hashes[1][None,:])
        # Same normalized source name is a poor negative even when addresses differ.
        name_groups = np.asarray([rows["name_group"][int(i)] for i in indices])
        duplicate |= name_groups[:,None] == name_groups[None,:]
        np.fill_diagonal(duplicate,False)
        optimizer.zero_grad(set_to_none=True)
        loss = gradient_cache(model,tokens,a.microbatch,torch.as_tensor(duplicate,device="cuda"))
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        if not torch.isfinite(norm):
            raise FloatingPointError("Nonfinite gradient")
        optimizer.step(); scheduler.step()
        if step==first_step or (step+1)%20==0:
            progress={"step":step+1,"total_steps":len(schedule),"loss":loss,"country":country,
                      "elapsed_seconds":time.monotonic()-started,
                      "gpu_peak_gb":torch.cuda.max_memory_allocated()/1e9}
            print(json.dumps(progress),flush=True)
            (a.output / "progress.json").write_text(json.dumps(progress,indent=2)+"\n")
        if (step+1)%250==0:
            save(step+1)
    save(len(schedule),final=True)
    config.update(elapsed_seconds=time.monotonic()-started,completed_steps=len(schedule))
    (a.output / "complete.json").write_text(json.dumps(config,indent=2)+"\n")


if __name__ == "__main__":
    main()
