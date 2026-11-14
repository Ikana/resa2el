# Bootstrap intervals (95%, 2000 draws of the mentions, seeds averaged first)

## Validation (16692 mentions): ReSa2 minus uniform, in points

| Measure | Difference | 95% interval |
|---|---|---|
| top-1 | +5.21 | +4.85 to +5.56 |
| recall at 10 | +1.11 | +0.91 to +1.29 |
| recall at 100 | +0.13 | +0.04 to +0.23 |

## New York Times (769 mentions): each trained sampler minus the released model, in points

| Sampler | top-1 | recall at 10 | recall at 100 |
|---|---|---|---|
| in_batch | -4.6 (-6.5 to -2.8) | -9.1 (-11.5 to -6.8) | -12.6 (-15.0 to -10.0) |
| uniform | -3.2 (-5.4 to -1.3) | -6.1 (-9.3 to -2.9) | -6.6 (-9.8 to -3.4) |
| topscore | -8.5 (-10.7 to -6.1) | -19.2 (-22.7 to -15.9) | -30.5 (-34.2 to -26.7) |
| resa2 | -1.5 (-3.6 to +0.6) | -2.6 (-5.5 to +0.4) | -1.4 (-4.2 to +1.6) |

## What ReSa2 stage 1 keeps (200 validation pools, 20 draws each)

| Measure | Share |
|---|---|
| Kept pages that are among the 500 closest in score to the correct page | 96% |
| Kept pages shared by two draws | 95% |
| Pages after stage 2 that are among the 75 most similar pages of the whole pool | 89% |
| Pages after stage 2 shared by two draws | 97% |
