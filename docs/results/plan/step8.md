# Four new samplers (2026-11-16)

7 negatives for each mention, 1,500 updates, same setup as the other runs. All numbers in percent.
Validation has 16,692 mentions, NYT has 769. A cell with ± is the mean and the standard deviation over the seeds.
Intervals are 95% bootstrap intervals over mentions, with the seeds averaged first.

| Run | valid top-1 | valid R@10 | valid R@100 | NYT top-1 | NYT R@10 | NYT R@100 |
|---|---|---|---|---|---|---|
| released | 42.6 | 77.7 | 93.1 | 13.0 | 33.6 | 57.3 |
| resa2 (3 seeds) | 69.5 ± 0.4 | 93.3 ± 0.1 | 98.5 ± 0.1 | 11.5 ± 0.2 | 30.9 ± 1.9 | 55.9 ± 0.5 |
| all pool pages, 999 negatives | 76.0 | 94.7 | 98.8 | 10.3 | 31.7 | 56.6 |
| score-proportional with correction | 75.7 | 94.6 | 98.8 | 11.1 | 29.3 | 55.5 |
| mixed hard and random | 69.4 | 92.9 | 98.4 | 10.8 | 26.7 | 50.8 |
| live top below the correct page | 73.4 | 94.4 | 98.7 | 11.1 | 30.2 | 56.0 |
| live top, seed 0 | 75.7 | 94.7 | 98.7 | 11.3 | 31.5 | 56.4 |
| live top (3 seeds) | 75.6 ± 0.0 | 94.7 ± 0.1 | 98.7 ± 0.0 | 10.6 ± 0.6 | 30.0 ± 1.4 | 54.8 ± 1.4 |
| resa2 mixed 50% (3 seeds) | 62.4 ± 0.3 | 91.0 ± 0.0 | 97.9 ± 0.1 | 13.4 ± 0.6 | 37.0 ± 0.9 | 61.6 ± 0.8 |
| live top mixed 50% (seeds 1, 2) | 67.6 ± 0.3 | 92.4 ± 0.0 | 98.1 ± 0.0 | 12.4 ± 0.5 | 36.3 ± 0.4 | 61.1 ± 0.3 |

## Differences (first minus second, percentage points, with interval)

| Difference | valid top-1 | valid R@10 | NYT R@10 | NYT R@100 |
|---|---|---|---|---|
| live top (3 seeds) − resa2 (3 seeds) | +6.07 (+5.71 to +6.43) | +1.33 (+1.16 to +1.52) | -0.87 (-2.38 to +0.65) | -1.08 (-2.60 to +0.39) |
| live top (3 seeds) − all pool pages, 999 negatives | -0.36 (-0.65 to -0.07) | -0.04 (-0.19 to +0.11) | -1.69 (-3.25 to -0.22) | -1.73 (-2.99 to -0.43) |
| score-proportional with correction − resa2 (3 seeds) | +6.17 (+5.76 to +6.59) | +1.26 (+1.05 to +1.47) | -1.65 (-3.55 to +0.30) | -0.39 (-2.30 to +1.56) |
| score-proportional with correction − live top, seed 0 | +0.06 (-0.24 to +0.37) | -0.07 (-0.23 to +0.10) | -2.21 (-3.77 to -0.65) | -0.91 (-2.47 to +0.65) |
| live top below the correct page − live top, seed 0 | -2.23 (-2.65 to -1.82) | -0.26 (-0.47 to -0.04) | -1.30 (-3.51 to +0.78) | -0.39 (-2.21 to +1.43) |
| mixed hard and random − resa2 (3 seeds) | -0.10 (-0.52 to +0.31) | -0.47 (-0.70 to -0.24) | -4.25 (-6.20 to -2.43) | -5.07 (-7.20 to -3.03) |
| live top (3 seeds) − released | +33.03 (+32.21 to +33.79) | +16.94 (+16.39 to +17.51) | -3.51 (-6.37 to -0.52) | -2.51 (-5.59 to +0.48) |
| live top mixed 50% (seeds 1, 2) − released | +24.98 (+24.31 to +25.65) | +14.70 (+14.20 to +15.23) | +2.73 (+0.65 to +4.88) | +3.71 (+1.63 to +5.85) |
| live top mixed 50% (seeds 1, 2) − resa2 mixed 50% (3 seeds) | +5.13 (+4.79 to +5.48) | +1.37 (+1.19 to +1.55) | -0.69 (-1.95 to +0.63) | -0.54 (-1.84 to +0.63) |
