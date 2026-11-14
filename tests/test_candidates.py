import numpy as np
import pytest

from resa2el import candidates as C
from resa2el import samplers as S

M, N = 40, 1000


@pytest.fixture
def pools():
    rng = np.random.default_rng(0)
    ids = np.stack([rng.choice(50000, N, replace=False) for _ in range(M)])
    gold = ids[np.arange(M), rng.integers(0, 30, M)].copy()
    gold[::7] = 60000 + np.arange(len(gold[::7]))  # some correct pages are not in their pool
    return {"ids": ids, "scores": rng.normal(-5, 3, (M, N)).astype(np.float32), "gold": gold,
            "gold_score": np.zeros(M, np.float32), "sims": rng.uniform(0, 1, (M, N)).astype(np.float32)}


@pytest.mark.parametrize("name,k,kw", [("uniform", 7, {}), ("uniform", 63, {}), ("topscore", 7, {}), ("resa2", 7, {}),
                                       ("stage2_only", 7, {}), ("band", 7, {"lo": 75, "hi": 300}), ("band", 7, {"lo": 300, "hi": None}),
                                       ("allpool", 999, {})])
def test_pool_samplers_give_gold_first_and_k_distinct_pool_pages(pools, name, k, kw):
    idx = np.arange(M)
    cand = C.batch_candidates(idx, pools, name, k, np.random.default_rng(1), kw)
    assert cand.shape == (M, k + 1)
    for r in idx:
        assert cand[r, 0] == pools["gold"][r]
        neg = cand[r, 1:].tolist()
        assert len(set(neg)) == k and pools["gold"][r] not in neg
        assert set(neg) <= set(pools["ids"][r].tolist())


def test_gold_random_uses_correct_pages_of_other_mentions(pools):
    gold_ids = pools["gold"]
    for k in (7, 30):
        cand = C.batch_candidates(np.arange(M), pools, "gold_random", k, np.random.default_rng(2), gold_ids=gold_ids)
        for r in range(M):
            neg = cand[r, 1:].tolist()
            assert len(set(neg)) == k and cand[r, 0] not in neg and set(neg) <= set(gold_ids.tolist())


def test_heldout_is_fixed_and_removed_from_training():
    keep = np.arange(0, 5000, 3)
    tr1, h1 = C.split_heldout(keep, 128)
    tr2, h2 = C.split_heldout(keep, 128)
    assert np.array_equal(h1, h2) and len(h1) == 128 and len(tr1) == len(keep) - 128
    assert not set(tr1.tolist()) & set(h1.tolist())
    tr, h = C.split_heldout(keep, 0)
    assert len(h) == 0 and np.array_equal(tr, keep)
