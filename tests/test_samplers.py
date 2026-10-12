import numpy as np
import pytest

from resa2el import samplers as S

N, K = 1000, 7


@pytest.fixture
def pool():
    rng = np.random.default_rng(0)
    scores = rng.normal(-4.0, 2.5, N)
    sims = rng.uniform(0, 1, N)
    return scores, 0.0, sims


@pytest.mark.parametrize("name", list(S.SAMPLERS))
def test_returns_k_distinct_valid_indices(name, pool):
    scores, s_pos, sims = pool
    sel = np.asarray(S.SAMPLERS[name](scores, s_pos, sims, K, np.random.default_rng(1)))
    assert sel.shape == (K,)
    assert len(set(sel.tolist())) == K
    assert sel.min() >= 0 and sel.max() < N


@pytest.mark.parametrize("name", list(S.SAMPLERS))
def test_same_seed_same_result(name, pool):
    scores, s_pos, sims = pool
    a = S.SAMPLERS[name](scores, s_pos, sims, K, np.random.default_rng(5))
    b = S.SAMPLERS[name](scores, s_pos, sims, K, np.random.default_rng(5))
    assert np.array_equal(np.sort(a), np.sort(b))


def test_topscore_takes_highest(pool):
    scores, s_pos, sims = pool
    sel = S.topscore(scores, s_pos, sims, K, np.random.default_rng(0))
    assert set(np.asarray(sel).tolist()) == set(np.argsort(-scores)[:K].tolist())


def test_stage1_weight_is_one_at_positive_score():
    w = S.stage1_weights(np.array([0.0, 2.0, -2.0]), 0.0)
    assert np.allclose(w, [1.0, np.exp(-1.0), np.exp(-1.0)])


def test_uniform_inclusion_is_k_over_n():
    """Each page has P(selected) = k/n. Check on a small pool."""
    n, k, reps = 20, 3, 20000
    rng = np.random.default_rng(2)
    scores, sims = np.zeros(n), np.zeros(n)
    counts = np.zeros(n)
    for _ in range(reps):
        counts[np.asarray(S.uniform(scores, 0.0, sims, k, rng))] += 1
    assert np.allclose(counts / reps, k / n, atol=0.015)


def test_resa2_selects_only_from_top_by_similarity():
    """With n_stage1 = pool size, stage 1 keeps every page, so stage 2
    can select only the n_stage2 pages with the highest similarity."""
    n = 50
    rng = np.random.default_rng(3)
    scores, sims = rng.normal(0, 1, n), rng.uniform(0, 1, n)
    top = set(np.argsort(-sims)[:10].tolist())
    for _ in range(200):
        sel = S.resa2(scores, 0.0, sims, 3, rng, n_stage1=n, n_stage2=10)
        assert set(np.asarray(sel).tolist()) <= top
