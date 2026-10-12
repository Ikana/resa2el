/-!
# Small Lean model of the false-negative claim in the proposal

Core Lean 4 only (no Mathlib). Check with:  cd lean && lake build

Setup. A page is its score (an integer, to stay simple). The correct page has
score `sPos`. A "possible false negative" is a page that scores above `sPos`.
A sampler returns a list of pages taken from the pool.

Main result (`topscore_is_max`): the top-score rule selects `min k A` possible
false negatives, where `A` is how many the pool has, and NO sampler of `k`
pages can select more (`any_sampler_le`). So "ReSa2 selects fewer of them than
top-score, in expectation" is true for every sampler, every pool, and every
run. It needs no condition on ReSa2.
-/

/-- `s` scores above the correct page. -/
def above (sPos s : Int) : Bool := decide (sPos < s)

/-- Number of possible false negatives in a list of pages. -/
def nAbove (sPos : Int) (l : List Int) : Nat := l.countP (above sPos)

/-! ## 1. An upper bound for every sampler -/

/-- Any selection taken from the pool (each page used at most once) has at most
`min (its size) A` possible false negatives. (Write the selected pages in
pool order; then "each page used at most once" means `sel` is a sublist.) -/
theorem any_sampler_le (sPos : Int) (pool sel : List Int) (h : sel.Sublist pool) :
    nAbove sPos sel ≤ min sel.length (nAbove sPos pool) :=
  Nat.le_min.mpr ⟨List.countP_le_length, h.countP_le⟩

/-! ## 2. The top-score rule reaches the bound

Sorting by score puts every page above `sPos` before every page below it.
So, for counting, top-score is: pages above first, then the others, take `k`. -/

def topscore (sPos : Int) (k : Nat) (pool : List Int) : List Int :=
  (pool.filter (above sPos) ++ pool.filter (fun s => !above sPos s)).take k

/-- Taking `k` from (all-good list ++ no-good list) gives `min k (good length)` goods. -/
theorem countP_take_append (p : Int → Bool) (a b : List Int)
    (ha : ∀ x ∈ a, p x = true) (hb : ∀ x ∈ b, p x = false) :
    ∀ k, ((a ++ b).take k).countP p = min k a.length := by
  induction a with
  | nil =>
    intro k
    simp only [List.nil_append, List.length_nil, Nat.min_zero]
    exact List.countP_eq_zero.mpr fun x hx => by
      simp [hb x (List.mem_of_mem_take hx)]
  | cons x xs ih =>
    intro k
    cases k with
    | zero => simp
    | succ k =>
      have hx : p x = true := ha x (by simp)
      have ih' := ih (fun y hy => ha y (by simp [hy])) k
      simp only [List.cons_append, List.take_succ_cons, List.countP_cons, hx,
        ite_true, List.length_cons, ih']
      omega

theorem topscore_is_max (sPos : Int) (k : Nat) (pool : List Int) :
    nAbove sPos (topscore sPos k pool) = min k (nAbove sPos pool) := by
  unfold nAbove topscore
  rw [countP_take_append (above sPos) _ _ (fun x hx => (List.mem_filter.mp hx).2)
      (fun x hx => by simpa using (List.mem_filter.mp hx).2)]
  rw [List.countP_eq_length_filter]

/-! ## 3. Stage 2 of ReSa2 picks 7 of the top 75 uniformly

Counting fact: of all 7-page subsets of 75 pages, the share that contains one
fixed page is 7/75. So P(page selected) = (7/75) · P(page is in the top 75),
which is at most (7/75) · P(page is kept by stage 1). -/

/-- Binomial coefficient. Step i: C(n, i+1) = C(n, i) · (n − i) / (i + 1),
    and each division is exact. -/
def choose (n k : Nat) : Nat :=
  (List.range k).foldl (fun c i => c * (n - i) / (i + 1)) 1

/-- Subsets that contain the fixed page: choose the other 6 from 74.
    `75 · C(74,6) = 7 · C(75,7)`, that is, the share is 7/75. -/
theorem stage2_share : 75 * choose 74 6 = 7 * choose 75 7 := by decide

/-! ## Example with numbers -/

-- Pool: correct page at 0; three pages above it.
#eval nAbove 0 [5, -3, 1, -8, 2, -1]                 -- 3
#eval topscore 0 2 [5, -3, 1, -8, 2, -1]             -- [5, 1]
#eval nAbove 0 (topscore 0 7 [5, -3, 1, -8, 2, -1])  -- 3 = min 7 3
