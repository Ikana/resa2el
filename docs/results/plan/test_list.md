# Step 7: test set, fixed list

Written on 2026-11-15, before any model is evaluated on the test set.
Approved by Rodrigo on 2026-11-15 ("run it!").

The test set is the Wikipedia test split (`out/test.jsonl`). Each model below is evaluated on it one time.
No other model is evaluated on the test set. No model or setting is changed after these numbers are seen.

| Row of the main table | Models | Evaluations |
|---|---|---|
| Released EVELINK | `models/biencoder` | 1 |
| Uniform from the pool, 7 negatives | `runs/main_lr1e-5/uniform_s0`, `_s1`, `_s2` | 3 |
| ReSa2, 7 negatives | `runs/main_lr1e-5/resa2_s0`, `_s1`, `_s2` | 3 |
| Stage 2 only, 7 negatives | `runs/plan/stage2only_s0`, `_s1`, `_s2` | 3 |
| All pool pages (about 999 negatives) | `runs/plan/allpool_s0` | 1 |
| ReSa2 mixed 50% with the released model | mix of `resa2_s0`, `_s1`, `_s2` with alpha 0.5, each alone | 3 |
| Total | | 14 |

Rows of the main table that are not in this list: in-batch, top-score, correct pages of other mentions (7 and 63), uniform with 63 negatives.
The models of the last three were deleted after the validation evaluation (disk space). These rows get no test number.

Metrics: top-1, recall at 10, recall at 100. For rows with 3 seeds: mean and standard deviation over the seeds.
Script: `env/run_test.sh`. Output: `runs/plan/ranks_test/NAME.npz` on the instance.
