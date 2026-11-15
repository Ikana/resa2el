import numpy as np

from resa2el.bandit import Exp3


def test_probs_are_a_distribution_with_a_floor():
    b = Exp3(4, 150)
    b.w[:] = [50.0, 0.0, 0.0, 0.0]
    p = b.probs()
    assert abs(p.sum() - 1) < 1e-12
    assert p.min() >= b.gamma / 4 - 1e-12


def test_scale_is_between_0_and_1_and_uses_percentiles():
    b = Exp3(4, 150)
    assert b.scale(3.0) == 0.5  # no history yet
    b.history = list(np.linspace(0, 1, 101))
    assert b.scale(-5) == 0.0 and b.scale(5) == 1.0
    assert abs(b.scale(0.5) - 0.5) < 1e-9
    assert abs(b.scale(0.2) - 0.0) < 1e-9 and abs(b.scale(0.8) - 1.0) < 1e-9


def test_only_the_picked_arm_changes():
    b = Exp3(4, 150)
    b.history = [0.0, 1.0, 0.2, 0.8, 0.5]
    b.update(2, 0.9)
    assert b.w[2] > 0 and (b.w[[0, 1, 3]] == 0).all()


def test_finds_a_much_better_arm():
    rng = np.random.default_rng(0)
    b = Exp3(4, 1500)
    for _ in range(1500):
        a = b.pick(rng)
        b.update(a, rng.normal(1.0 if a == 1 else 0.0, 0.5))
    assert b.probs().argmax() == 1 and b.probs()[1] > 0.5
