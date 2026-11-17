# Second test round: results

Wikipedia test split, 19267 mentions. The list was fixed in `test_list2.md` before the evaluations.
Round 1 rows are copied from `step7_test.md` and were not run again.

| Row | Top-1 | R@10 | R@100 |
|---|---|---|---|
| Released EVELINK (round 1) | 43.0 | 77.5 | 93.4 |
| ReSa2 (round 1) | 71.1 ± 0.4 | 93.8 ± 0.0 | 98.6 ± 0.0 |
| All pool pages (round 1) | 77.7 | 95.2 | 98.9 |
| ReSa2 mixed 50% (round 1) | 63.0 ± 0.4 | 91.5 ± 0.2 | 98.0 ± 0.1 |
| Live hard negatives (round 2) | 77.7 ± 0.3 | 95.0 ± 0.0 | 98.9 ± 0.0 |
| Live hard negatives mixed 50% (round 2) | 69.0 ± 0.2 | 92.9 ± 0.0 | 98.3 ± 0.0 |

## Comparisons fixed in the list

Difference in points, paired bootstrap over mentions (10000 resamples), mean over the seeds first.

| Comparison | Metric | Difference | 95% interval | 99% interval |
|---|---|---|---|---|
| Live hard negatives minus ReSa2 | Top-1 | +6.56 | +6.23 to +6.91 | +6.12 to +7.01 |
| Live hard negatives minus ReSa2 | R@10 | +1.20 | +1.04 to +1.36 | +0.99 to +1.41 |
| Live hard negatives mixed 50% minus ReSa2 mixed 50% | Top-1 | +5.99 | +5.67 to +6.31 | +5.58 to +6.42 |
| Live hard negatives mixed 50% minus ReSa2 mixed 50% | R@10 | +1.42 | +1.25 to +1.60 | +1.19 to +1.65 |
| Live hard negatives minus Released EVELINK | Top-1 | +34.72 | +34.00 to +35.44 | +33.78 to +35.65 |
| Live hard negatives minus Released EVELINK | R@10 | +17.52 | +16.98 to +18.07 | +16.83 to +18.25 |
| Live hard negatives mixed 50% minus Released EVELINK | Top-1 | +26.07 | +25.45 to +26.70 | +25.25 to +26.90 |
| Live hard negatives mixed 50% minus Released EVELINK | R@10 | +15.38 | +14.88 to +15.88 | +14.71 to +16.05 |

## The models trained again: validation and NYT

| Seed | Valid top-1 | Valid R@10 | Valid R@100 | NYT top-1 | NYT R@10 | NYT R@100 |
|---|---|---|---|---|---|---|
| 0 | 75.6 | 94.7 | 98.7 | 11.3 | 31.5 | 56.4 |
| 1 | 75.6 | 94.6 | 98.7 | 10.3 | 28.7 | 54.1 |
| 2 | 75.6 | 94.7 | 98.8 | 10.3 | 29.9 | 54.0 |
| Mean | 75.6 ± 0.0 | 94.7 ± 0.1 | 98.7 ± 0.0 | 10.6 ± 0.6 | 30.0 ± 1.4 | 54.8 ± 1.4 |

First training of the same seeds (`step8.md`): validation top-1 75.6 ± 0.0, R@10 94.7 ± 0.1; NYT R@10 30.0 ± 1.4.

## Reading

- The claim fixed in the list holds: live hard negatives is above ReSa2 at top-1 on the test set, and the 99% interval is above zero.
- The test set gives no information about news text. The NYT numbers have no second data set.
- This was the second use of the test set. The first list (`test_list.md`) said that there is one use only.
