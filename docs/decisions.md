# Decisions

Newest at the bottom. One line for the decision, one line for the reason.

- 2026-10-12: Two Python environments. Sampler code uses modern Python (3.10+, numpy only).
  EVELINK training uses its own pinned environment (Python 3.8, torch 1.6) on the GPU machine.
  Reason: the samplers must be testable on any laptop; EVELINK's old versions are hard to change.
- 2026-10-12: Samplers share one signature `(scores, s_pos, sims, k, rng)`.
  Reason: the training code can swap samplers with one config value.
- 2026-10-12: Lean proofs use core Lean 4 only (no Mathlib) for now.
  Reason: proofs so far are about lists and counting. Add Mathlib when a proof needs real numbers.
- 2026-10-12: Every run reads one config file in `configs/` and writes to `runs/<name>/<seed>/`.
  Reason: all twelve runs must be repeatable and must use the same budget.
