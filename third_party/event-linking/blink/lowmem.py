"""Low-memory mode for EVELINK prediction (added for resa2el).

The original code loads three large Wikipedia files into Python dicts
(about 55 GB of RAM) and keeps the full candidate encoding
(5.9M x 1024 floats) on the GPU. This module gives the same results
with much less memory:

1. `build_store` runs once, on a machine with a lot of RAM. It writes the
   per-candidate title, text (first 2000 characters) and hyperlinks into
   one SQLite file, with exactly the logic of `EvaluateBLINK`. It also
   writes the candidate encoding as a plain .npy file.
2. `LowMemStore` reads that SQLite file one candidate at a time, so only
   the top-k candidates of each mention are read.
3. `chunked_topk` finds the bi-encoder top-k by streaming the candidate
   encoding through the GPU in chunks, from a memory-mapped .npy file.
"""
import json
import os
import sqlite3

import numpy as np
import torch

STORE_FILE = "entities.sqlite"
ENCODING_FILE = "all_entities.npy"


def build_store(id2title, title_text, t2h, out_dir):
    """Write the candidate maps of EvaluateBLINK to SQLite.

    The loop body is a copy of the loop in event_linking_main.EvaluateBLINK,
    so local ids, texts and hyperlinks are the same as in the original code.
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, STORE_FILE)
    if os.path.exists(path):
        os.remove(path)
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE ent (id INTEGER PRIMARY KEY, title TEXT, text TEXT, hyper TEXT)"
    )
    rows = []
    idx = 0
    for title in title_text:
        text = title_text[title][0:2000]
        hyper = []

        title_form = title
        if title[0] == "'" and title[-1] == "'":
            title_form = title[1:-1]

        if title_form.replace(" ", '_') in id2title[1] and id2title[1][title_form.replace(" ", '_')] in t2h:
            for t_idx, linked_title in enumerate(t2h[id2title[1][title_form.replace(" ", '_')]]):
                if t_idx > 10:
                    break
                link = t2h[id2title[1][title_form.replace(" ", '_')]][linked_title]
                if link['end'] > 2000:
                    continue
                hyper.append((linked_title, title_text[title][link['start']: link['end']]))

        rows.append((idx, title, text, json.dumps(hyper)))
        if len(rows) >= 100000:
            conn.executemany("INSERT INTO ent VALUES (?, ?, ?, ?)", rows)
            rows = []
        idx += 1
    if rows:
        conn.executemany("INSERT INTO ent VALUES (?, ?, ?, ?)", rows)
    conn.commit()
    conn.close()
    return idx


def convert_encoding(encoding_path, out_dir):
    """Save the candidate encoding as a plain .npy file (can be memory-mapped)."""
    enc = torch.load(encoding_path, map_location="cpu", weights_only=False)
    arr = enc.numpy()
    np.save(os.path.join(out_dir, ENCODING_FILE), arr)
    return arr.shape, str(arr.dtype)


class _Column:
    """Read-only dict-like view of one column of the store."""

    def __init__(self, store, column, decode=None):
        self.store = store
        self.column = column
        self.decode = decode

    def __getitem__(self, key):
        row = self.store.conn.execute(
            "SELECT {} FROM ent WHERE id = ?".format(self.column), (int(key),)
        ).fetchone()
        if row is None:
            raise KeyError(key)
        value = row[0]
        return self.decode(value) if self.decode else value

    def __contains__(self, key):
        # The original maps have an entry for every local id.
        return 0 <= int(key) < len(self.store)

    def __len__(self):
        return len(self.store)


class LowMemStore:
    def __init__(self, out_dir):
        path = os.path.join(out_dir, STORE_FILE)
        self.conn = sqlite3.connect("file:{}?mode=ro".format(path), uri=True)
        self.size = self.conn.execute("SELECT COUNT(*) FROM ent").fetchone()[0]
        self.title = _Column(self, "title")
        self.text = _Column(self, "text")
        self.hyper = _Column(self, "hyper", decode=lambda s: [tuple(x) for x in json.loads(s)])

    def __len__(self):
        return self.size


def load_encoding(out_dir):
    return np.load(os.path.join(out_dir, ENCODING_FILE), mmap_mode="r")


def chunked_topk(context_emb, cand_enc, top_k, device="cuda", cand_chunk=500000, ctx_chunk=2048):
    """Top-k of context_emb @ cand_enc.T without holding cand_enc on the GPU.

    context_emb: float tensor N x d (CPU). cand_enc: array M x d (memmap).
    Returns (scores, indices) as numpy arrays N x top_k, best first.
    """
    n = context_emb.size(0)
    ctx = context_emb.to(device)
    best_s = torch.full((n, top_k), float("-inf"), device=device)
    best_i = torch.zeros((n, top_k), dtype=torch.long, device=device)
    m = cand_enc.shape[0]
    for start in range(0, m, cand_chunk):
        end = min(start + cand_chunk, m)
        chunk = torch.from_numpy(np.ascontiguousarray(cand_enc[start:end])).to(device)
        if chunk.dtype != ctx.dtype:
            chunk = chunk.to(ctx.dtype)
        for c0 in range(0, n, ctx_chunk):
            c1 = min(c0 + ctx_chunk, n)
            scores = ctx[c0:c1].mm(chunk.t())
            k = min(top_k, end - start)
            s, i = scores.topk(k)
            all_s = torch.cat([best_s[c0:c1], s], dim=1)
            all_i = torch.cat([best_i[c0:c1], i + start], dim=1)
            s2, pos = all_s.topk(top_k)
            best_s[c0:c1] = s2
            best_i[c0:c1] = all_i.gather(1, pos)
            del scores, s, i, all_s, all_i
        del chunk
        print("chunked_topk: {}/{} candidates".format(end, m), flush=True)
    return best_s.cpu().numpy(), best_i.cpu().numpy()
