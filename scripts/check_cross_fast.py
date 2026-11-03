"""Check the fast crossencoder path of EVELINK (added for resa2el).

Run from the EVELINK folder:  python ../../scripts/check_cross_fast.py [--real]
Prints CROSS_FAST_CHECK=PASS when all checks pass.

Checks:
1. _modify_fast gives the same tensor as train_cross.modify.
2. LowMemStore can be read from a worker thread (prefetch).
3. _score_in_chunks gives the same result with and without prefetch, with
   and without checkpoints, and continues a stopped run from its checkpoints.
4. _ckpt_meta refuses a checkpoint directory from another run.
5. (--real) The released crossencoder gives the same ranking in float32 and
   in bfloat16 autocast on the CPU, on a few random inputs.
"""
import json
import os
import shutil
import sys
import tempfile
import threading

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
from blink import lowmem  # noqa: E402
from blink import main_dense  # noqa: E402
from blink.crossencoder.train_cross import evaluate, modify  # noqa: E402


def check_modify_fast():
    rng = np.random.default_rng(0)
    for n, k, ctx_len, cand_len, max_len in [(3, 5, 8, 6, 13), (2, 4, 8, 6, 20), (4, 3, 256, 256, 512), (1, 2, 7, 9, 10)]:
        ctx = torch.from_numpy(rng.integers(0, 30000, size=(n, ctx_len)))
        cand = torch.from_numpy(rng.integers(0, 30000, size=(n, k, cand_len)))
        want = modify(ctx, cand, max_len)
        got = main_dense._modify_fast(ctx, cand, max_len)
        assert want.shape == got.shape, (want.shape, got.shape)
        assert torch.equal(want, got), (n, k, ctx_len, cand_len, max_len)
        assert got.dtype == torch.long
    print("modify_fast: ok")


def tiny_store(d):
    title_text, t2h, t2id = {}, {}, {}
    for i in range(50):
        title = "Page {}".format(i)
        title_text[title] = "text of page {} ".format(i) * 20
        t2id[title.replace(" ", "_")] = str(100 + i)
        t2h[str(100 + i)] = {"Link {}".format(i): {"start": 0, "end": 4}}
    n = lowmem.build_store(({}, t2id), title_text, t2h, d)
    assert n == 50
    return lowmem.LowMemStore(d)


def check_store_threads():
    with tempfile.TemporaryDirectory() as d:
        store = tiny_store(d)
        main = [(store.title[i], store.text[i], store.hyper[i]) for i in range(50)]
        got = []
        errors = []

        def worker():
            try:
                got.extend((store.title[i], store.text[i], store.hyper[i]) for i in range(50))
            except Exception as e:  # noqa: BLE001
                errors.append(repr(e))

        t = threading.Thread(target=worker)
        t.start()
        t.join()
        assert not errors, errors
        assert got == main
        assert store.hyper[3] == [("Link 3", "text")]
        assert 49 in store.title and 50 not in store.title
    print("store threads: ok")


def fake_prep(c0, c1):
    return np.arange(c0, c1)


def fake_score(prepared):
    scores = np.stack([np.sin(prepared * j + 0.1) for j in range(1, 6)], axis=1)
    return np.argsort(scores, axis=1), scores


def check_score_in_chunks():
    n = 23
    want_index, want_scores = fake_score(fake_prep(0, n))
    for prefetch in (False, True):
        for chunk in (5, 23, 50):
            index, scores = main_dense._score_in_chunks(n, chunk, fake_prep, fake_score, ckpt_dir=None, prefetch=prefetch)
            assert np.array_equal(index, want_index) and np.array_equal(scores, want_scores), (prefetch, chunk)
    # checkpoints: a run that stops in chunk 3 continues from the saved chunks
    with tempfile.TemporaryDirectory() as d:
        calls = []

        def stop_at_third(prepared):
            calls.append(int(prepared[0]))
            if len(calls) == 3:
                raise RuntimeError("stopped")
            return fake_score(prepared)

        try:
            main_dense._score_in_chunks(n, 5, fake_prep, stop_at_third, ckpt_dir=d, prefetch=True)
            raise AssertionError("expected RuntimeError")
        except RuntimeError:
            pass
        saved = sorted(f for f in os.listdir(d) if f.startswith("cross_") and f.endswith(".npz") and ".tmp" not in f)
        assert saved == ["cross_00000000_00000005.npz", "cross_00000005_00000010.npz"], saved
        assert not [f for f in os.listdir(d) if ".tmp" in f], os.listdir(d)
        calls2 = []

        def count(prepared):
            calls2.append(int(prepared[0]))
            return fake_score(prepared)

        index, scores = main_dense._score_in_chunks(n, 5, fake_prep, count, ckpt_dir=d, prefetch=True)
        assert calls2 == [10, 15, 20], calls2
        assert np.array_equal(index, want_index) and np.array_equal(scores, want_scores)
        # a third run reads everything from the checkpoints
        index, scores = main_dense._score_in_chunks(n, 5, fake_prep, count, ckpt_dir=d, prefetch=True)
        assert calls2 == [10, 15, 20], calls2
        assert np.array_equal(index, want_index) and np.array_equal(scores, want_scores)
        # chunks scored with another precision are refused
        try:
            main_dense._score_in_chunks(n, 5, fake_prep, count, ckpt_dir=d, prefetch=True, tag="amp=fp16")
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
    # a chunk file that appears during the run does not shift the prefetch order
    with tempfile.TemporaryDirectory() as d:
        def prep_and_plant(c0, c1):
            if c0 == 5:
                np.savez(os.path.join(d, "cross_00000010_00000015.npz"), index=np.zeros((5, 5), int), scores=np.zeros((5, 5)), tag="")
            return fake_prep(c0, c1)

        index, scores = main_dense._score_in_chunks(n, 5, prep_and_plant, fake_score, ckpt_dir=d, prefetch=True)
        assert np.array_equal(index, want_index) and np.array_equal(scores, want_scores)
        saved = np.load(os.path.join(d, "cross_00000010_00000015.npz"))
        assert np.array_equal(saved["scores"], want_scores[10:15])
    print("score_in_chunks: ok")


def check_ckpt_meta():
    with tempfile.TemporaryDirectory() as d:
        meta = {"n": 3, "top_k": 100, "chunk": 500, "first": "a", "last": "b"}
        main_dense._ckpt_meta(d, meta)
        main_dense._ckpt_meta(d, dict(meta))
        try:
            main_dense._ckpt_meta(d, dict(meta, n=4))
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
    print("ckpt_meta: ok")


class _Loader:
    """Minimal stand-in for the dataloader in evaluate()."""

    def __init__(self, batches):
        self.batches = batches

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


def check_real_model():
    from blink.crossencoder.crossencoder import load_crossencoder
    with open("models/crossencoder/params.json") as f:
        params = json.load(f)
    params["path_to_model"] = None  # the checkpoint was saved on a GPU: load it on the CPU below
    params["no_cuda"] = True
    params["data_parallel"] = False  # no DataParallel wrapper on the CPU
    reranker = load_crossencoder(params)
    reranker.load_model("models/crossencoder/pytorch_model.bin", cpu=True)
    reranker.model.eval()
    rng = np.random.default_rng(1)
    n, k, ctx_len = 2, 4, params["max_context_length"]
    ctx = rng.integers(1000, 20000, size=(n, ctx_len))
    cand = rng.integers(1000, 20000, size=(n, k, params["max_cand_length"]))
    cand[:, :, 0] = reranker.tokenizer.cls_token_id
    x = main_dense._modify_fast(torch.from_numpy(ctx), torch.from_numpy(cand), params["max_seq_length"])
    labels = torch.zeros(n, dtype=torch.long)
    loader = _Loader([(x, labels)])
    full = np.asarray(evaluate(reranker, loader, "cpu", None, ctx_len, silent=True)["logits"])
    amp = np.asarray(evaluate(reranker, loader, "cpu", None, ctx_len, silent=True, amp_dtype=torch.bfloat16)["logits"])
    assert full.shape == amp.shape == (n, k), (full.shape, amp.shape)
    assert full.dtype == np.float32 and amp.dtype == np.float32
    print("real model logits float32:", full.round(3).tolist())
    print("real model logits bf16 autocast:", amp.round(3).tolist())
    print("max abs diff: {:.4f}, top-1 agree: {}/{}".format(
        np.abs(full - amp).max(), int((full.argmax(1) == amp.argmax(1)).sum()), n))
    assert np.abs(full - amp).max() < 1.0
    assert (full.argmax(1) == amp.argmax(1)).all()


if __name__ == "__main__":
    check_modify_fast()
    check_store_threads()
    check_score_in_chunks()
    check_ckpt_meta()
    if "--real" in sys.argv:
        check_real_model()
    print("CROSS_FAST_CHECK=PASS")
