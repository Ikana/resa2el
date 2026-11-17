import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Data.Fintype.Card

/-!
# Why the choice of negatives changes what one training step teaches

Uses Mathlib. Check with: `cd lean && lake build`.

Setup. One mention has a correct page with score `sPos` and a pool of wrong
pages `i : ι` with scores `s i`. The weight of page `i` is

  w i = e^{-(sPos - s i)} = e^{s i - sPos}.

A sampler selects a set `S` of pool pages. Its mass is `M(S) = Σ_{i ∈ S} w i`.
The loss on `S` is `log (1 + M(S))` (see `LossBound.lean`). The total push of
the step on the mention vector is `M(S) / (1 + M(S))`.

Results.
* `push_eq_one_sub_exp_neg_loss`: push = 1 - e^{-loss}.
* `mass_mono`, `lossOn_mono`, `push_mono`: a larger selection has more mass,
  more loss and more push. So the whole pool is an upper limit.
* `sum_le_sum_top`: among all selections of the same size, the pages with the
  highest weights have the most mass. So "top k by current score" has the
  largest loss and push that k negatives can have (`lossOn_le_top`,
  `push_le_top`).
* `top_ge_uniform_share`: the top `k` pages carry at least the share `k / N`
  of the mass of all `N` pool pages. `k / N` is the expected share of `k`
  uniform picks.
-/

open Finset

variable {ι : Type*}

/-- Weight of pool page `i`: `e^{s i - sPos}`. -/
noncomputable def weight (sPos : ℝ) (s : ι → ℝ) (i : ι) : ℝ := Real.exp (-(sPos - s i))

/-- Mass of a selection: the sum of the weights of its pages. -/
noncomputable def mass (sPos : ℝ) (s : ι → ℝ) (S : Finset ι) : ℝ := ∑ i ∈ S, weight sPos s i

/-- Loss of the step when the negatives are the pages of `S`. -/
noncomputable def lossOn (sPos : ℝ) (s : ι → ℝ) (S : Finset ι) : ℝ := Real.log (1 + mass sPos s S)

/-- Total push of the step: the sum of the shares of the negatives. -/
noncomputable def push (sPos : ℝ) (s : ι → ℝ) (S : Finset ι) : ℝ := mass sPos s S / (1 + mass sPos s S)

theorem weight_pos (sPos : ℝ) (s : ι → ℝ) (i : ι) : 0 < weight sPos s i := Real.exp_pos _

theorem mass_nonneg (sPos : ℝ) (s : ι → ℝ) (S : Finset ι) : 0 ≤ mass sPos s S :=
  Finset.sum_nonneg fun i _ => (weight_pos sPos s i).le

/-! ## 1. Push and loss are the same quantity on two scales -/

/-- push = 1 - e^{-loss}. -/
theorem push_eq_one_sub_exp_neg_loss (sPos : ℝ) (s : ι → ℝ) (S : Finset ι) :
    push sPos s S = 1 - Real.exp (-lossOn sPos s S) := by
  have h : 0 < 1 + mass sPos s S := by linarith [mass_nonneg sPos s S]
  rw [push, lossOn, Real.exp_neg, Real.exp_log h]
  field_simp
  ring

/-- The push grows with the mass. -/
theorem div_one_add_mono {a b : ℝ} (ha : 0 ≤ a) (hab : a ≤ b) : a / (1 + a) ≤ b / (1 + b) := by
  rw [div_le_div_iff₀ (by linarith) (by linarith)]
  nlinarith

/-! ## 2. More negatives: more mass, more loss, more push -/

theorem mass_mono (sPos : ℝ) (s : ι → ℝ) {S T : Finset ι} (h : S ⊆ T) :
    mass sPos s S ≤ mass sPos s T :=
  Finset.sum_le_sum_of_subset_of_nonneg h fun i _ _ => (weight_pos sPos s i).le

theorem lossOn_mono (sPos : ℝ) (s : ι → ℝ) {S T : Finset ι} (h : S ⊆ T) :
    lossOn sPos s S ≤ lossOn sPos s T :=
  Real.log_le_log (by linarith [mass_nonneg sPos s S]) (by linarith [mass_mono sPos s h])

theorem push_mono (sPos : ℝ) (s : ι → ℝ) {S T : Finset ι} (h : S ⊆ T) :
    push sPos s S ≤ push sPos s T :=
  div_one_add_mono (mass_nonneg sPos s S) (mass_mono sPos s h)

/-! ## 3. With `k` negatives, the top `k` by weight are the best choice

`T` is a top set with threshold `θ` when every page in `T` has weight at least
`θ` and every page outside `T` has weight at most `θ`. Sorting by score and
taking the first `k` pages gives such a set. -/

def TopSet (w : ι → ℝ) (T : Finset ι) (θ : ℝ) : Prop :=
  (∀ i ∈ T, θ ≤ w i) ∧ (∀ i, i ∉ T → w i ≤ θ)

/-- Any selection of the same size has at most the sum of a top set. -/
theorem sum_le_sum_top [DecidableEq ι] (w : ι → ℝ) (S T : Finset ι) (θ : ℝ) (hT : TopSet w T θ)
    (hc : S.card = T.card) : ∑ i ∈ S, w i ≤ ∑ i ∈ T, w i := by
  have hS : ∑ i ∈ S \ T, w i + ∑ i ∈ S ∩ T, w i = ∑ i ∈ S, w i := by
    have := Finset.sum_sdiff (f := w) (Finset.inter_subset_left : S ∩ T ⊆ S)
    rwa [Finset.sdiff_inter_self_left] at this
  have hT' : ∑ i ∈ T \ S, w i + ∑ i ∈ T ∩ S, w i = ∑ i ∈ T, w i := by
    have := Finset.sum_sdiff (f := w) (Finset.inter_subset_left : T ∩ S ⊆ T)
    rwa [Finset.sdiff_inter_self_left] at this
  have hcard : (S \ T).card = (T \ S).card := Finset.card_sdiff_comm hc
  have h1 : ∑ i ∈ S \ T, w i ≤ (S \ T).card • θ :=
    Finset.sum_le_card_nsmul _ _ _ fun i hi => hT.2 i (Finset.mem_sdiff.mp hi).2
  have h2 : (T \ S).card • θ ≤ ∑ i ∈ T \ S, w i :=
    Finset.card_nsmul_le_sum _ _ _ fun i hi => hT.1 i (Finset.mem_sdiff.mp hi).1
  rw [hcard] at h1
  rw [Finset.inter_comm T S] at hT'
  linarith

/-- A top set has the most mass among selections of its size. -/
theorem mass_le_top [DecidableEq ι] (sPos : ℝ) (s : ι → ℝ) (S T : Finset ι) (θ : ℝ)
    (hT : TopSet (weight sPos s) T θ) (hc : S.card = T.card) :
    mass sPos s S ≤ mass sPos s T :=
  sum_le_sum_top (weight sPos s) S T θ hT hc

theorem lossOn_le_top [DecidableEq ι] (sPos : ℝ) (s : ι → ℝ) (S T : Finset ι) (θ : ℝ)
    (hT : TopSet (weight sPos s) T θ) (hc : S.card = T.card) :
    lossOn sPos s S ≤ lossOn sPos s T :=
  Real.log_le_log (by linarith [mass_nonneg sPos s S])
    (by linarith [mass_le_top sPos s S T θ hT hc])

theorem push_le_top [DecidableEq ι] (sPos : ℝ) (s : ι → ℝ) (S T : Finset ι) (θ : ℝ)
    (hT : TopSet (weight sPos s) T θ) (hc : S.card = T.card) :
    push sPos s S ≤ push sPos s T :=
  div_one_add_mono (mass_nonneg sPos s S) (mass_le_top sPos s S T θ hT hc)

/-! ## 4. The top `k` carry at least the share `k / N` of the whole pool -/

/-- `k · (sum over the pool) ≤ N · (sum over a top set of size k)`. -/
theorem top_ge_uniform_share [Fintype ι] [DecidableEq ι] (w : ι → ℝ) (T : Finset ι) (θ : ℝ) (hT : TopSet w T θ) :
    (T.card : ℝ) * ∑ i, w i ≤ (Fintype.card ι : ℝ) * ∑ i ∈ T, w i := by
  have hsplit : ∑ i ∈ T, w i + ∑ i ∈ Tᶜ, w i = ∑ i, w i := Finset.sum_add_sum_compl T w
  have hN : (T.card : ℝ) + (Tᶜ.card : ℝ) = (Fintype.card ι : ℝ) := by
    exact_mod_cast Finset.card_add_card_compl T
  have h1 : ∑ i ∈ Tᶜ, w i ≤ (Tᶜ.card : ℝ) * θ := by
    have := Finset.sum_le_card_nsmul Tᶜ w θ fun i hi => hT.2 i (Finset.mem_compl.mp hi)
    simpa [nsmul_eq_mul] using this
  have h2 : (T.card : ℝ) * θ ≤ ∑ i ∈ T, w i := by
    have := Finset.card_nsmul_le_sum T w θ fun i hi => hT.1 i hi
    simpa [nsmul_eq_mul] using this
  have hk : (0 : ℝ) ≤ T.card := Nat.cast_nonneg _
  have hc : (0 : ℝ) ≤ Tᶜ.card := Nat.cast_nonneg _
  have e1 := mul_le_mul_of_nonneg_left h1 hk
  have e2 := mul_le_mul_of_nonneg_left h2 hc
  rw [← hN, ← hsplit]
  nlinarith [e1, e2]
