"""Check that EVELINK low-memory mode gives the same results as the original code.

Run from the EVELINK folder:  python ../path/to/check_lowmem.py
Prints LOWMEM_CHECK=PASS when all checks pass.
"""
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
from blink import lowmem  # noqa: E402


def original_maps(id2title, title_text, t2h):
    # Copy of the loop in event_linking_main.EvaluateBLINK.
    localid2title, localid2text, localid2hyperlinks = {}, {}, {}
    idx = 0
    for title in title_text:
        localid2title[idx] = title
        localid2text[idx] = title_text[title][0:2000]
        localid2hyperlinks[idx] = []
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
                localid2hyperlinks[idx].append((linked_title, title_text[title][link['start']: link['end']]))
        idx += 1
    return localid2title, localid2text, localid2hyperlinks


def check_store():
    rng = np.random.default_rng(0)
    title_text, t2h, t2id = {}, {}, {}
    for i in range(300):
        title = "'Page {}'".format(i) if i % 7 == 0 else "Page {}".format(i)
        text = "".join(rng.choice(list("abc def ghi ")) for _ in range(int(rng.integers(10, 3000))))
        title_text[title] = text
        plain = title[1:-1] if title.startswith("'") else title
        if i % 5:
            t2id[plain.replace(" ", "_")] = str(1000 + i)
            links = {}
            for j in range(int(rng.integers(0, 15))):
                s = int(rng.integers(0, 2500))
                links["Link {}".format(j)] = {"start": s, "end": s + int(rng.integers(1, 30))}
            t2h[str(1000 + i)] = links
    id2title = ({}, t2id)
    want = original_maps(id2title, title_text, t2h)
    with tempfile.TemporaryDirectory() as d:
        n = lowmem.build_store(id2title, title_text, t2h, d)
        store = lowmem.LowMemStore(d)
        assert n == len(store) == len(want[0])
        for i in range(n):
            assert store.title[i] == want[0][i]
            assert store.text[i] == want[1][i]
            assert store.hyper[i] == want[2][i], (i, store.hyper[i], want[2][i])
            assert np.int64(i) in store.hyper
    print("store check ok")


def check_topk(device):
    g = torch.Generator().manual_seed(0)
    ctx = torch.randn(700, 64, generator=g)
    cand = torch.randn(23456, 64, generator=g).numpy()
    want_s, want_i = ctx.to(device).mm(torch.from_numpy(cand).to(device).t()).topk(100)
    got_s, got_i = lowmem.chunked_topk(ctx, cand, 100, device=device, cand_chunk=5000, ctx_chunk=128)
    assert np.allclose(got_s, want_s.cpu().numpy(), atol=1e-4)
    same = (got_i == want_i.cpu().numpy()).mean()
    assert same > 0.999, same
    print("topk check ok, same indices:", same)


if __name__ == "__main__":
    check_store()
    check_topk("cuda" if torch.cuda.is_available() else "cpu")
    print("LOWMEM_CHECK=PASS")
