# Plan steps 4 and 6: similarity bands and stage 2 only

All numbers in percent. Validation has 16,692 mentions, NYT has 769. Intervals are 95% bootstrap intervals over mentions.

| Run | valid top-1 | valid R@10 | valid R@100 | NYT top-1 | NYT R@10 | NYT R@100 |
|---|---|---|---|---|---|---|
| released | 42.6 | 77.7 | 93.1 | 13.0 | 33.6 | 57.3 |
| resa2 (3 seeds) | 69.5 | 93.3 | 98.5 | 11.5 | 30.9 | 55.9 |
| stage2only (3 seeds) | 69.2 | 92.9 | 98.3 | 10.1 | 27.0 | 49.7 |
| band top 25 | 69.5 | 92.7 | 98.3 | 11.2 | 26.7 | 50.5 |
| band 25 to 75 | 66.6 | 92.8 | 98.4 | 10.0 | 26.8 | 49.9 |
| band 75 to 300 | 63.3 | 92.4 | 98.3 | 9.9 | 28.3 | 49.7 |
| band rest | 59.9 | 91.7 | 98.3 | 9.6 | 28.5 | 51.5 |

## Differences

| A minus B | split | top-1 | R@10 | R@100 |
|---|---|---|---|---|
| stage2only (3 seeds) minus resa2 (3 seeds) | valid | -0.30 (-0.62 to +0.02) | -0.42 (-0.58 to -0.25) | -0.19 (-0.28 to -0.10) |
| stage2only (3 seeds) minus resa2 (3 seeds) | NYT | -1.43 (-2.47 to -0.39) | -3.90 (-5.33 to -2.47) | -6.20 (-7.67 to -4.72) |
| band top 25 minus band 25 to 75 | valid | +2.94 (+2.40 to +3.49) | -0.04 (-0.32 to +0.24) | -0.05 (-0.21 to +0.08) |
| band top 25 minus band 25 to 75 | NYT | +1.17 (-0.39 to +2.73) | -0.13 (-2.21 to +1.95) | +0.52 (-1.82 to +2.73) |
| band 25 to 75 minus band 75 to 300 | valid | +3.24 (+2.68 to +3.77) | +0.35 (+0.05 to +0.66) | +0.04 (-0.09 to +0.17) |
| band 25 to 75 minus band 75 to 300 | NYT | +0.13 (-1.30 to +1.56) | -1.56 (-3.64 to +0.52) | +0.26 (-1.95 to +2.47) |
| band 75 to 300 minus band rest | valid | +3.43 (+2.84 to +4.03) | +0.70 (+0.39 to +1.03) | +0.02 (-0.14 to +0.17) |
| band 75 to 300 minus band rest | NYT | +0.26 (-1.17 to +1.82) | -0.13 (-2.21 to +1.95) | -1.82 (-4.03 to +0.26) |
| band top 25 minus band rest | valid | +9.60 (+8.93 to +10.27) | +1.01 (+0.66 to +1.37) | +0.01 (-0.16 to +0.17) |
| band top 25 minus band rest | NYT | +1.56 (-0.13 to +3.25) | -1.82 (-4.29 to +0.65) | -1.04 (-3.90 to +1.56) |
| band top 25 minus resa2 (3 seeds) | valid | -0.03 (-0.49 to +0.41) | -0.61 (-0.85 to -0.37) | -0.22 (-0.34 to -0.09) |
| band top 25 minus resa2 (3 seeds) | NYT | -0.35 (-1.60 to +0.91) | -4.25 (-6.24 to -2.12) | -5.46 (-7.72 to -3.25) |
| stage2only (3 seeds) minus released | valid | +26.66 (+25.92 to +27.40) | +15.19 (+14.62 to +15.76) | +5.21 (+4.86 to +5.55) |
| stage2only (3 seeds) minus released | NYT | -2.90 (-5.03 to -0.82) | -6.55 (-9.67 to -3.38) | -7.63 (-10.66 to -4.51) |
| band rest minus released | valid | +17.33 (+16.58 to +18.11) | +13.99 (+13.41 to +14.59) | +5.18 (+4.84 to +5.54) |
| band rest minus released | NYT | -3.38 (-5.59 to -1.17) | -5.07 (-8.32 to -1.69) | -5.85 (-9.24 to -2.47) |

## Seeds

| Run | valid top-1 | valid R@10 | valid R@100 | NYT top-1 | NYT R@10 | NYT R@100 |
|---|---|---|---|---|---|---|
| stage2only_s0 | 69.6 | 92.8 | 98.3 | 10.3 | 26.9 | 50.1 |
| stage2only_s1 | 69.3 | 92.7 | 98.3 | 9.4 | 27.4 | 47.6 |
| stage2only_s2 | 68.8 | 93.2 | 98.4 | 10.7 | 26.7 | 51.5 |
| resa2_s0 | 70.0 | 93.2 | 98.4 | 11.7 | 28.7 | 55.4 |
| resa2_s1 | 69.4 | 93.3 | 98.5 | 11.4 | 32.1 | 56.3 |
| resa2_s2 | 69.3 | 93.4 | 98.6 | 11.4 | 31.9 | 56.0 |

## Reward log (step 4)

Held-out loss: cross entropy against the full pool on 128 held-out training mentions, measured each 10 updates. The reward of an interval is the fall of this loss.

| Run | loss at 0 | at 100 | at 300 | at 750 | at 1,500 | lowest (update) | mean reward 0 to 300 | mean reward 300 to 1,500 | SD of reward 300 to 1,500 |
|---|---|---|---|---|---|---|---|---|---|
| band_25to75_s0 | 2.942 | 1.663 | 1.609 | 1.444 | 1.135 | 1.093 (1320) | +0.0445 | +0.0039 | 0.0399 |
| band_75to300_s0 | 2.942 | 1.922 | 1.835 | 1.440 | 1.142 | 1.111 (1010) | +0.0369 | +0.0058 | 0.0429 |
| band_rest_s0 | 2.942 | 2.032 | 2.026 | 1.758 | 1.370 | 1.370 (1500) | +0.0306 | +0.0055 | 0.0518 |
| band_top25_s0 | 2.942 | 1.616 | 1.402 | 1.114 | 0.797 | 0.748 (1430) | +0.0513 | +0.0050 | 0.0356 |
