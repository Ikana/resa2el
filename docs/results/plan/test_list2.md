# Second test round: fixed list

Written on 2026-11-16, before any model of this list is evaluated on the test set.
Approved by Rodrigo on 2026-11-16 ("Yeah, sure. Try another data set that we've never seen before, just as a final validation. And we'll call it in the paper what it is.").

## Why there is a second round

`test_list.md` (2026-11-15) says "No other model is evaluated on the test set". This list breaks that sentence, and the paper must say so.
After the first round I tried four new samplers and chose one of them (live hard negatives, `--sampler live_top`) with validation and NYT numbers.
The test set had no part in that choice. This round is the confirmation of that choice on data that was not used to make it.

## Rules

- Each model below is evaluated on the Wikipedia test split (`out/test.jsonl`) one time.
- No other model is evaluated. No third round follows.
- No model, sampler or mixing weight is changed after these numbers are seen. The numbers are reported whatever they are.
- The first-round numbers (`step7_test.md`) are not run again. They are the comparison.

## The list

| Row | Models | Evaluations |
|---|---|---|
| Live hard negatives, 7 negatives | `runs/test2/livetop_s0`, `_s1`, `_s2` | 3 |
| Live hard negatives mixed 50% with the released model | mix of `livetop_s1`, `_s2` with alpha 0.5, each alone | 2 |
| Total | | 5 |

The 50% mix has seeds 1 and 2 only, the same as on validation (`step8.md`).

## The models are trained again

The models of `step8.md` were deleted after their validation evaluation (disk space). The same command and seeds train them again.
Each new model is evaluated on validation and NYT first (`runs/test2/ranks/`), so that a difference from the first training is visible.
The two validation results are both reported.

## Comparisons, fixed now

1. Live hard negatives against ReSa2 (first round, 3 seeds): top-1 and recall at 10. Paired bootstrap over mentions, mean over the seeds first, 95% and 99% intervals.
2. Live hard negatives mixed 50% (2 seeds) against ReSa2 mixed 50% (first round, 3 seeds): the same.
3. Each of the two rows against the released model: the same.

Claim to confirm: the validation gain of live hard negatives over ReSa2 at top-1 (+6.1 points) holds on the test set, that is the 99% interval is above zero.

Script: `env/run_test2.sh`. Output: `runs/test2/ranks_test/NAME.npz` on the instance.
