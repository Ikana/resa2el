# Plan steps 1 and 5: weight mixing and the bandit

Mix: weights = (1 - a) x released + a x ReSa2 model, one mix for each of the 3 ReSa2 seeds; cells are the mean of the 3 mixes.
The 50% mix was named as the main result before any of these numbers was seen.
All numbers in percent. Validation has 16,692 mentions, NYT has 769. Intervals are 95% bootstrap intervals over mentions (seeds averaged first).

| Model | valid top-1 | valid R@10 | valid R@100 | NYT top-1 | NYT R@10 | NYT R@100 |
|---|---|---|---|---|---|---|
| released (0%) | 42.6 | 77.7 | 93.1 | 13.0 | 33.6 | 57.3 |
| mix 25% | 53.8 | 86.7 | 96.7 | 13.8 | 37.0 | 60.1 |
| mix 50% (main) | 62.4 | 91.0 | 97.9 | 13.4 | 37.0 | 61.6 |
| mix 75% | 67.4 | 92.9 | 98.4 | 12.3 | 34.1 | 59.9 |
| resa2 (100%) | 69.5 | 93.3 | 98.5 | 11.5 | 30.9 | 55.9 |
| average of the 3 resa2 models | 71.0 | 94.0 | 98.7 | 12.4 | 31.6 | 56.7 |
| all pool pages | 76.0 | 94.7 | 98.8 | 10.3 | 31.7 | 56.6 |
| bandit | 68.5 | 93.2 | 98.4 | 9.9 | 29.0 | 52.1 |

## Differences

| A minus B | split | top-1 | R@10 | R@100 |
|---|---|---|---|---|
| mix 50% (main) minus released (0%) | valid | +19.85 (+19.21 to +20.46) | +13.33 (+12.79 to +13.84) | +4.80 (+4.48 to +5.10) |
| mix 50% (main) minus released (0%) | NYT | +0.35 (-1.21 to +1.91) | +3.42 (+1.39 to +5.46) | +4.25 (+2.17 to +6.29) |
| mix 50% (main) minus resa2 (100%) | valid | -7.11 (-7.59 to -6.68) | -2.28 (-2.52 to -2.04) | -0.60 (-0.71 to -0.48) |
| mix 50% (main) minus resa2 (100%) | NYT | +1.82 (+0.43 to +3.30) | +6.07 (+3.90 to +8.15) | +5.68 (+3.81 to +7.67) |
| mix 75% minus released (0%) | valid | +24.83 (+24.13 to +25.51) | +15.18 (+14.62 to +15.76) | +5.28 (+4.94 to +5.61) |
| mix 75% minus released (0%) | NYT | -0.74 (-2.60 to +1.08) | +0.52 (-2.04 to +2.90) | +2.60 (-0.09 to +5.29) |
| mix 75% minus resa2 (100%) | valid | -2.13 (-2.41 to -1.86) | -0.44 (-0.57 to -0.29) | -0.11 (-0.18 to -0.05) |
| mix 75% minus resa2 (100%) | NYT | +0.74 (-0.22 to +1.69) | +3.16 (+1.82 to +4.46) | +4.03 (+2.90 to +5.24) |
| mix 25% minus released (0%) | valid | +11.25 (+10.75 to +11.77) | +8.93 (+8.53 to +9.33) | +3.56 (+3.30 to +3.85) |
| mix 25% minus released (0%) | NYT | +0.78 (-0.39 to +1.95) | +3.47 (+1.86 to +5.07) | +2.73 (+1.21 to +4.29) |
| average of the 3 resa2 models minus resa2 (100%) | valid | +1.46 (+1.20 to +1.73) | +0.65 (+0.51 to +0.79) | +0.17 (+0.10 to +0.24) |
| average of the 3 resa2 models minus resa2 (100%) | NYT | +0.82 (-0.04 to +1.60) | +0.69 (-0.39 to +1.78) | +0.78 (-0.35 to +2.04) |
| average of the 3 resa2 models minus released (0%) | valid | +28.42 (+27.60 to +29.17) | +16.26 (+15.68 to +16.83) | +5.57 (+5.21 to +5.92) |
| average of the 3 resa2 models minus released (0%) | NYT | -0.65 (-2.86 to +1.69) | -1.95 (-5.07 to +1.04) | -0.65 (-3.77 to +2.47) |
| bandit minus resa2 (100%) | valid | -1.02 (-1.44 to -0.59) | -0.08 (-0.31 to +0.14) | -0.09 (-0.21 to +0.03) |
| bandit minus resa2 (100%) | NYT | -1.65 (-2.95 to -0.35) | -1.91 (-3.90 to +0.22) | -3.77 (-5.68 to -1.82) |

## Bandit (EXP3 over the 4 bands, 150 rounds of 10 updates)

| Band | picks, rounds 1 to 50 | 51 to 100 | 101 to 150 | all | mean scaled reward | probability at the end |
|---|---|---|---|---|---|---|
| top25 | 12 | 11 | 14 | 37 | 0.34 | 0.25 |
| 25to75 | 14 | 15 | 19 | 48 | 0.37 | 0.35 |
| 75to300 | 14 | 13 | 13 | 40 | 0.33 | 0.26 |
| rest | 10 | 11 | 4 | 25 | 0.30 | 0.14 |

Highest probability of any band in any round: 0.36. Held-out loss at the end: 0.910.
