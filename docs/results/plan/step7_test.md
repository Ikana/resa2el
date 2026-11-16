# Step 7: test set results

Wikipedia test split. Each model was evaluated one time (list fixed in `test_list.md`).
Mean over the seeds, with the standard deviation over the seeds (n - 1) where there are 3 seeds.

| Row | Seeds | Top-1 | R@10 | R@100 |
|---|---|---|---|---|
| Released EVELINK | 1 | 43.0 | 77.5 | 93.4 |
| Uniform from the pool, 7 negatives | 3 | 65.5 ± 0.8 | 92.8 ± 0.1 | 98.6 ± 0.1 |
| ReSa2, 7 negatives | 3 | 71.1 ± 0.4 | 93.8 ± 0.0 | 98.6 ± 0.0 |
| Stage 2 only, 7 negatives | 3 | 71.0 ± 0.2 | 93.4 ± 0.2 | 98.5 ± 0.1 |
| All pool pages (about 999 negatives) | 1 | 77.7 | 95.2 | 98.9 |
| ReSa2 mixed 50% with the released model | 3 | 63.0 ± 0.4 | 91.5 ± 0.2 | 98.0 ± 0.1 |

Mentions: 19267.
