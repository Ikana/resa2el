# Main runs (2026-11-04/05)

4 samplers x 3 seeds, 1,500 updates each, batch 32 (4 micro-batches of 8), lr 3e-5 with 10% warmup,
page encoder frozen, 7 negatives per mention from a fixed pool of 1,000 (released bi-encoder scores).
Recall on the validation set (16,692 mentions) against all 5.87M pages. Released model: 42.6 / 77.7 / 93.1.

| Sampler | seed 0 (top-1 / R@10 / R@100) | seed 1 | seed 2 | mean top-1 | mean R@10 | mean R@100 |
|---|---|---|---|---|---|---|
| in_batch | 24.4 / 57.6 / 82.9 | 22.7 / 56.4 / 83.1 | 25.0 / 60.7 / 85.0 | 24.0 | 58.2 | 83.6 |
| uniform | 60.7 / 90.0 / 98.0 | 61.5 / 90.3 / 97.9 | 61.1 / 90.4 / 97.8 | 61.1 | 90.3 | 97.9 |
| topscore | 55.2 / 78.3 / 90.1 | 0.0 / 0.1 / 0.9 (collapsed) | 57.8 / 80.6 / 92.3 | 37.7 | 53.0 | 61.1 |
| resa2 | 68.3 / 92.0 / 98.0 | 0.0 / 0.1 / 1.2 (collapsed) | 67.5 / 92.1 / 98.1 | 45.3 | 61.4 | 65.8 |

Healthy seeds only: uniform 61.1 / 90.3 / 97.9; topscore 56.5 / 79.4 / 91.2; resa2 67.9 / 92.0 / 98.1.

Collapsed runs: the loss jumps once (resa2 seed 1 at update ~220, topscore seed 1 at ~400, near the end of
the warmup at the peak learning rate) and stays at about 0.27-0.30. The mention embeddings then score about 2
against every page (released model: about 550), so the ranking is noise. Weights are finite.
