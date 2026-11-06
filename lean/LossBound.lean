import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Algebra.BigOperators.Fin

/-!
# Easy negatives add almost nothing to the loss

Uses Mathlib for the real numbers, `Real.exp`, and `Real.log`. Check with:
`cd lean && lake build`.

Setup. One training mention has a correct page with score `sPos` and `n`
negatives with scores `s i`. The retriever loss (softmax cross entropy) is

  L = -log (e^{sPos} / (e^{sPos} + Σ e^{s i})).

The gap of negative `i` is `g i = sPos - s i`: how far it scores below the
correct page.

Results.
* `loss_eq_log_gaps`: L = log (1 + Σ e^{-g i}). The loss depends only on
  the gaps.
* `loss_le_sum_exp_neg_gap`: L ≤ Σ e^{-g i}, because log (1 + x) ≤ x.
  So a negative with gap g adds at most e^{-g} to the bound.
* `loss_nonneg`: L ≥ 0.
-/

open Real Finset

/-- The softmax cross-entropy loss of one mention: correct-page score `sPos`,
negative scores `s 0, …, s (n-1)`. -/
noncomputable def loss {n : ℕ} (sPos : ℝ) (s : Fin n → ℝ) : ℝ :=
  -Real.log (Real.exp sPos / (Real.exp sPos + ∑ i, Real.exp (s i)))

/-- The loss written with the gaps `sPos - s i`. -/
theorem loss_eq_log_gaps {n : ℕ} (sPos : ℝ) (s : Fin n → ℝ) :
    loss sPos s = Real.log (1 + ∑ i, Real.exp (-(sPos - s i))) := by
  have hpos : 0 < Real.exp sPos := Real.exp_pos sPos
  have hsum : 0 ≤ ∑ i, Real.exp (s i) :=
    Finset.sum_nonneg fun i _ => (Real.exp_pos (s i)).le
  -- e^{-(sPos - s i)} = e^{s i} / e^{sPos}
  have hgap : ∀ i, Real.exp (-(sPos - s i)) = Real.exp (s i) / Real.exp sPos := by
    intro i
    rw [neg_sub, Real.exp_sub]
  -- 1 + Σ e^{-g i} = (e^{sPos} + Σ e^{s i}) / e^{sPos}
  have hrw : 1 + ∑ i, Real.exp (-(sPos - s i))
      = (Real.exp sPos + ∑ i, Real.exp (s i)) / Real.exp sPos := by
    rw [eq_div_iff hpos.ne', add_mul, one_mul, Finset.sum_mul]
    congr 1
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [hgap i, div_mul_cancel₀ _ hpos.ne']
  rw [hrw, loss, ← Real.log_inv, inv_div]

/-- The loss is never negative. -/
theorem loss_nonneg {n : ℕ} (sPos : ℝ) (s : Fin n → ℝ) : 0 ≤ loss sPos s := by
  rw [loss_eq_log_gaps]
  apply Real.log_nonneg
  have : 0 ≤ ∑ i, Real.exp (-(sPos - s i)) :=
    Finset.sum_nonneg fun i _ => (Real.exp_pos _).le
  linarith

/-- Main bound: L ≤ Σ e^{-g i}. Each negative adds at most e^{-g} to it. -/
theorem loss_le_sum_exp_neg_gap {n : ℕ} (sPos : ℝ) (s : Fin n → ℝ) :
    loss sPos s ≤ ∑ i, Real.exp (-(sPos - s i)) := by
  rw [loss_eq_log_gaps]
  set x := ∑ i, Real.exp (-(sPos - s i)) with hx
  have hx0 : 0 ≤ x := Finset.sum_nonneg fun i _ => (Real.exp_pos _).le
  -- log y ≤ y - 1 for y > 0, with y = 1 + x
  have := Real.log_le_sub_one_of_pos (by linarith : (0 : ℝ) < 1 + x)
  linarith

/-- Corollary: if every negative scores at least `g` below the correct page,
then L ≤ n · e^{-g}. -/
theorem loss_le_of_gap_ge {n : ℕ} (sPos g : ℝ) (s : Fin n → ℝ)
    (hg : ∀ i, g ≤ sPos - s i) :
    loss sPos s ≤ n * Real.exp (-g) := by
  calc loss sPos s ≤ ∑ i, Real.exp (-(sPos - s i)) := loss_le_sum_exp_neg_gap sPos s
    _ ≤ ∑ _i : Fin n, Real.exp (-g) :=
        Finset.sum_le_sum fun i _ => Real.exp_le_exp.mpr (by linarith [hg i])
    _ = n * Real.exp (-g) := by simp
