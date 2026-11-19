import Mathlib.Tactic

/-!
# When the half-and-half mix of two models beats both

Uses Mathlib. Check with: `cd lean && lake build`.

Setup. `θ 0` is the released model and `θ 1` is the trained model. The model
`θ α = (1 - α) θ 0 + α θ 1` is on the line between them, and `α = 1/2` is the
half-and-half mix. Suppose that the loss on some data, along this line, is the
quadratic `f α = a α² + b α + c` with `a > 0`. Its lowest point is at
`α* = -b / (2 a)`.

Result (`half_beats_both_iff`): the mix has a lower loss than both ends,
`f (1/2) < f 0` and `f (1/2) < f 1`, exactly when `1/4 < α* < 3/4`.
-/

/-- The loss along the line between the two models. -/
def lineLoss (a b c α : ℝ) : ℝ := a * α ^ 2 + b * α + c

theorem half_beats_both_iff (a b c : ℝ) (ha : 0 < a) :
    (lineLoss a b c (1 / 2) < lineLoss a b c 0 ∧ lineLoss a b c (1 / 2) < lineLoss a b c 1)
      ↔ (1 / 4 < -b / (2 * a) ∧ -b / (2 * a) < 3 / 4) := by
  have h2a : 0 < 2 * a := by linarith
  unfold lineLoss
  rw [lt_div_iff₀ h2a, div_lt_iff₀ h2a]
  constructor <;> rintro ⟨h1, h2⟩ <;> constructor <;> nlinarith
