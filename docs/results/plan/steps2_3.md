# Steps 2 and 3: baseline repair and no sampling (1 seed each, 1,500 updates, learning rate 1e-5)

| Run | Negatives per mention | Validation top-1 / R@10 / R@100 | NYT top-1 / R@10 / R@100 |
|---|---|---|---|
| Released model | - | 42.6 / 77.7 / 93.1 | 13.0 / 33.6 / 57.3 |
| In-batch, original code (3 seeds) | 7 | 33.6 / 69.6 / 89.8 | 8.5 / 24.4 / 44.8 |
| Uniform from the pool (3 seeds) | 7 | 64.3 / 92.2 / 98.4 | 9.8 / 27.4 / 50.8 |
| Top-score (3 seeds) | 7 | 65.3 / 87.2 / 95.9 | 4.6 / 14.4 / 26.8 |
| ReSa2 (3 seeds) | 7 | 69.5 / 93.3 / 98.5 | 11.5 / 30.9 / 55.9 |
| Random correct pages of other mentions | 7 | 32.9 / 68.2 / 88.7 | 7.4 / 21.5 / 40.3 |
| Random correct pages of other mentions | 63 | 42.9 / 79.4 / 94.3 | 8.2 / 23.1 / 42.1 |
| Uniform from the pool | 63 | 71.0 / 94.1 / 98.8 | 10.0 / 29.4 / 52.4 |
| All pool pages, no sampling | 999 | 76.0 / 94.7 / 98.8 | 10.3 / 31.7 / 56.6 |

## Differences on validation, in points (95% bootstrap interval over mentions; ReSa2 is the mean of 3 seeds)

| Comparison | top-1 | recall at 10 |
|---|---|---|
| All pool pages minus ReSa2 | +6.4 (+6.0 to +6.8) | +1.4 (+1.2 to +1.6) |
| Uniform with 63 minus ReSa2 | +1.5 (+1.1 to +1.9) | +0.7 (+0.5 to +1.0) |
| All pool pages minus uniform with 63 | +4.9 (+4.5 to +5.4) | |

## NYT recall at 10 minus the released model, in points

- allpool_s0: -1.8 (-5.2 to +1.4)
- goldrand63_s0: -10.4 (-13.3 to -7.7)
- goldrand7_s0: -12.1 (-15.0 to -9.2)
- uniform63_s0: -4.2 (-7.2 to -0.8)
