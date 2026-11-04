"""Build the candidate pools for sampler training (added for resa2el).

Run on the GPU machine, from the EVELINK folder:
  python ../../scripts/build_pools.py --split train --pool 1000 --out out/pools

For every mention of out/<split>.jsonl it finds, with the released bi-encoder:
  ids     (N x pool) int32    local ids of the pool pages, best first
  scores  (N x pool) float32  released bi-encoder score of each pool page
  gold    (N,)       int32    local id of the correct page (-1 if not found)
  gold_score (N,)    float32  released score of the correct page
  sims    (N x pool) float16  dot product of the correct page and each pool page
and writes them to <out>/<split>.npz. It also writes <out>/all_entities_fp16.npy
once (the released page vectors in float16, 12 GB), which training reads.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.getcwd())
from blink import lowmem  # noqa: E402
from blink.biencoder.biencoder import load_biencoder, to_bert_input  # noqa: E402
from blink.main_dense import _process_biencoder_dataloader  # noqa: E402

FP16_FILE = "all_entities_fp16.npy"


def title_to_local(store):
    """Map 'page title' (spaces, no quotes) -> local id, from the store."""
    out = {}
    for local_id, title in store.conn.execute("SELECT id, title FROM ent"):
        if title[0] == "'" and title[-1] == "'":
            title = title[1:-1]
        out.setdefault(title, local_id)
    return out


def fp16_encoding(lowmem_dir, out_dir, rows_per_step=200000):
    path = os.path.join(out_dir, FP16_FILE)
    if os.path.exists(path):
        return np.load(path, mmap_mode="r")
    src = lowmem.load_encoding(lowmem_dir)
    dst = np.lib.format.open_memmap(path + ".tmp", mode="w+", dtype=np.float16, shape=src.shape)
    for r in tqdm(range(0, src.shape[0], rows_per_step), desc="fp16 encoding"):
        dst[r:r + rows_per_step] = src[r:r + rows_per_step].astype(np.float16)
    dst.flush()
    del dst
    os.replace(path + ".tmp", path)
    return np.load(path, mmap_mode="r")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--pool", type=int, default=1000)
    ap.add_argument("--out", default="out/pools")
    ap.add_argument("--lowmem", default="out/lowmem")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    with open("models/biencoder/params.json") as f:
        params = json.load(f)
    params["path_to_model"] = "models/biencoder/pytorch_model.bin"
    params["debug"] = False
    biencoder = load_biencoder(params)
    biencoder.model.eval()

    samples = []
    with open("out/{}.jsonl".format(args.split)) as f:
        for line in f:
            samples.append(json.loads(line))
            if args.limit and len(samples) >= args.limit:
                break
    print("{} mentions".format(len(samples)), flush=True)

    store = lowmem.LowMemStore(args.lowmem)
    t2l = title_to_local(store)
    gold = np.array([t2l.get(s["label_title"].replace("_", " "), -1) for s in samples], dtype=np.int32)
    print("gold found for {}/{} mentions ({:.0f}s)".format(int((gold >= 0).sum()), len(gold), time.time() - t0), flush=True)
    del t2l

    enc = fp16_encoding(args.lowmem, args.out)
    print("encoding", enc.shape, enc.dtype, "({:.0f}s)".format(time.time() - t0), flush=True)

    dataloader = _process_biencoder_dataloader(samples, biencoder.tokenizer, params)
    ctx = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="encode mentions"):
            context_input, _, _ = batch
            token_idx, segment_idx, mask = to_bert_input(context_input, biencoder.NULL_IDX)
            emb, _ = biencoder.model(token_idx.to(args.device), segment_idx.to(args.device), mask.to(args.device), None, None, None)
            ctx.append(emb.float().cpu())
    ctx = torch.cat(ctx)
    with torch.no_grad():
        scores, ids = lowmem.chunked_topk(ctx, enc, args.pool, device=args.device)
    ids = ids.astype(np.int32)
    print("pools done ({:.0f}s)".format(time.time() - t0), flush=True)

    # released score of the correct page, and its similarity to each pool page
    gold_score = np.full(len(samples), np.nan, dtype=np.float32)
    sims = np.zeros(ids.shape, dtype=np.float16)
    step = 256
    with torch.no_grad():
        for b in tqdm(range(0, len(samples), step), desc="sims"):
            g = gold[b:b + step]
            ok = g >= 0
            if not ok.any():
                continue
            g_enc = torch.from_numpy(enc[g[ok]].astype(np.float32)).to(args.device)
            gold_score[b:b + step][ok] = (ctx[b:b + step][ok].to(args.device) * g_enc).sum(1).cpu().numpy()
            pool_enc = torch.from_numpy(np.take(enc, ids[b:b + step][ok], axis=0).astype(np.float32)).to(args.device)
            sims[b:b + step][ok] = torch.einsum("bd,bpd->bp", g_enc, pool_enc).cpu().numpy().astype(np.float16)
    in_pool = np.array([g in set(row.tolist()) for g, row in zip(gold, ids)])
    print("correct page inside its pool: {}/{}".format(int(in_pool.sum()), len(gold)), flush=True)
    path = os.path.join(args.out, "{}.npz".format(args.split))
    np.savez(path, ids=ids, scores=scores.astype(np.float32), gold=gold, gold_score=gold_score, sims=sims)
    print("wrote", path, "({:.0f}s)".format(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
