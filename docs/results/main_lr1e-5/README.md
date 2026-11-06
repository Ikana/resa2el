# Main runs at learning rate 1e-5 (2026-11-05/06)

Same setup as `../main/` (4 samplers x 3 seeds, 1,500 updates, batch 32 as 4 x 8, 10% warm-up,
page encoder frozen, 7 negatives from a fixed pool of 1,000) with the learning rate at 1e-5
instead of 3e-5. No run collapsed. Validation recall against all 5.87M pages; released model 42.6 / 77.7 / 93.1.

| Sampler | seed 0 (top-1 / R@10 / R@100) | seed 1 | seed 2 | top-1 | R@10 | R@100 |
|---|---|---|---|---|---|---|
| in_batch | 36.0 / 71.7 / 91.2 | 31.4 / 67.4 / 88.1 | 33.4 / 69.7 / 90.1 | 33.6 ± 1.9 | 69.6 ± 1.8 | 89.8 ± 1.3 |
| uniform | 64.4 / 92.3 / 98.5 | 64.5 / 92.4 / 98.4 | 64.1 / 92.0 / 98.4 | 64.3 ± 0.2 | 92.2 ± 0.1 | 98.4 ± 0.1 |
| topscore | 64.4 / 86.5 / 95.6 | 64.6 / 86.7 / 95.4 | 67.0 / 88.5 / 96.7 | 65.3 ± 1.2 | 87.2 ± 0.9 | 95.9 ± 0.6 |
| resa2 | 70.0 / 93.2 / 98.4 | 69.4 / 93.3 / 98.5 | 69.3 / 93.4 / 98.6 | 69.5 ± 0.3 | 93.3 ± 0.1 | 98.5 ± 0.1 |

(± is the population standard deviation over the 3 seeds.)

Compared with lr 3e-5 (`../main/`): no collapse, and every sampler is better (resa2 92.0 -> 93.3 R@10 on
healthy seeds, uniform 90.3 -> 92.2, topscore 79.4 -> 87.2, in_batch 58.2 -> 69.6).
These are the numbers to use for the report. The test set is untouched.
