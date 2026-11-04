"""Recall of a bi-encoder on one split, against all 5.87M pages (added for resa2el).

Run on the GPU machine, from the EVELINK folder:
  python ../../scripts/eval_recall.py --split valid --pools out/pools/valid.npz \
      --model models/biencoder/pytorch_model.bin [--limit N] [--out runs/x/recall.json]

The gold local ids come from the pools file of the split. Prints top-1,
recall@10 and recall@100 over the mentions that have a gold page, and
writes them as JSON when --out is given.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="valid")
    ap.add_argument("--pools", required=True, help="<split>.npz with the gold local ids")
    ap.add_argument("--model", required=True, help="pytorch_model.bin of the bi-encoder")
    ap.add_argument("--encoding", default="out/pools/all_entities_fp16.npy")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    t0 = time.time()

    with open("models/biencoder/params.json") as f:
        params = json.load(f)
    params["path_to_model"] = args.model
    params["debug"] = False
    biencoder = load_biencoder(params)
    biencoder.model.eval()

    samples = []
    with open("out/{}.jsonl".format(args.split)) as f:
        for line in f:
            samples.append(json.loads(line))
            if args.limit and len(samples) >= args.limit:
                break
    gold = np.load(args.pools)["gold"][:len(samples)]
    assert len(gold) == len(samples), (len(gold), len(samples))

    dataloader = _process_biencoder_dataloader(samples, biencoder.tokenizer, params)
    ctx = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="encode mentions"):
            context_input, _, _ = batch
            token_idx, segment_idx, mask = to_bert_input(context_input, biencoder.NULL_IDX)
            emb, _ = biencoder.model(token_idx.to(args.device), segment_idx.to(args.device), mask.to(args.device), None, None, None)
            ctx.append(emb.float().cpu())
        enc = np.load(args.encoding, mmap_mode="r")
        _, ids = lowmem.chunked_topk(torch.cat(ctx), enc, 100, device=args.device)

    ok = gold >= 0
    hit = ids[ok] == gold[ok][:, None]
    res = {
        "split": args.split, "model": args.model, "mentions": int(ok.sum()),
        "top1": float(hit[:, :1].any(1).mean()),
        "recall@10": float(hit[:, :10].any(1).mean()),
        "recall@100": float(hit.any(1).mean()),
        "seconds": round(time.time() - t0),
    }
    print(json.dumps(res), flush=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
