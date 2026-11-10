"""Per-mention retrieval ranks of a bi-encoder on the validation and NYT sets (added for resa2el).

Run on the GPU machine, from the EVELINK folder:
  python ../../scripts/eval_ranks.py --model MODEL.bin --out runs/gaps/NAME.npz

Encodes the validation mentions (out/valid.jsonl) and the NYT test mentions
(data/nyt/nyt_test.jsonl), ranks all pages once for both, and saves for every
mention: the split, the local id of the correct page (-1 when it is not in the
store or the mention is Nil), the rank of the correct page among the top 100
(-1 when it is not there), and the local ids of the top 10 pages. The titles of
the top 10 and of the correct page are saved too, for the error analysis.
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
sys.path.insert(0, os.path.join(os.getcwd(), "..", "..", "scripts"))
from blink import lowmem  # noqa: E402
from blink.biencoder.biencoder import load_biencoder, to_bert_input  # noqa: E402
from blink.main_dense import _process_biencoder_dataloader  # noqa: E402
from build_pools import title_to_local  # noqa: E402


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--encoding", default="out/pools/all_entities_fp16.npy")
    ap.add_argument("--lowmem", default="out/lowmem")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    t0 = time.time()

    with open("models/biencoder/params.json") as f:
        params = json.load(f)
    params["path_to_model"] = args.model
    params["debug"] = False
    biencoder = load_biencoder(params)
    biencoder.model.eval()

    valid = read_jsonl("out/valid.jsonl")
    nyt = read_jsonl("data/nyt/nyt_test.jsonl")
    # type and status of the validation mentions live in the preprocessed
    # clusters, in the same order as out/valid.jsonl (see data_processing.py)
    with open("data/wikipedia/preprocessed/wiki_valid.json") as f:
        clusters = json.load(f)
    flat = [m for title in clusters for m in clusters[title]]
    assert len(flat) == len(valid), (len(flat), len(valid))
    for s, m in zip(valid, flat):
        s["type"], s["status"] = m.get("type", ""), m.get("status", "")
    samples = valid + nyt
    split = np.array([0] * len(valid) + [1] * len(nyt), dtype=np.int8)

    store = lowmem.LowMemStore(args.lowmem)
    t2l = title_to_local(store)
    gold = np.array([t2l.get(str(s.get("label_title", "")).replace("_", " "), -1) for s in samples], dtype=np.int32)
    print("mentions {} (valid {}, nyt {}); gold found {} ({:.0f}s)".format(
        len(samples), len(valid), len(nyt), int((gold >= 0).sum()), time.time() - t0), flush=True)

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

    hit = ids == gold[:, None]
    rank = np.where(hit.any(1), hit.argmax(1), -1).astype(np.int16)
    top10 = ids[:, :10].astype(np.int32)
    titles10 = np.array([[store.title[int(i)] for i in row] for row in top10], dtype=object)
    gold_title = np.array([store.title[int(g)] if g >= 0 else "" for g in gold], dtype=object)
    ok = gold >= 0
    for name, m in (("valid", split == 0), ("nyt", split == 1)):
        mm = m & ok
        print("{}: n={} top1={:.3f} R@10={:.3f} R@100={:.3f}".format(
            name, int(mm.sum()), float((rank[mm] == 0).mean()), float(((rank[mm] >= 0) & (rank[mm] < 10)).mean()), float((rank[mm] >= 0).mean())), flush=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, split=split, gold=gold, rank=rank, top10=top10, titles10=titles10, gold_title=gold_title,
             mtype=np.array([s.get("type", "") for s in samples], dtype=object),
             status=np.array([s.get("status", "") for s in samples], dtype=object),
             model=np.array(args.model))
    print("wrote", args.out, "({:.0f}s)".format(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
