# Per-class recall, error types, and NYT (lr 1e-5 runs, mean ± sd over seeds)

## Recall@10 by mention class (validation)

| Model | Verb, seen | Verb, unseen form | Verb, unseen | Nominal, hard | Nominal, easy | All |
|---|---|---|---|---|---|---|
| Released | 76.1 | 77.3 | 82.9 | 68.9 | 82.6 | 77.7 |
| in_batch | 65.2 ± 1.5 | 69.0 ± 1.3 | 75.0 ± 0.6 | 58.0 ± 1.5 | 78.4 ± 3.9 | 69.6 ± 1.8 |
| uniform | 98.8 ± 0.2 | 91.3 ± 0.2 | 88.3 ± 0.1 | 88.8 ± 0.2 | 97.1 ± 0.3 | 92.2 ± 0.1 |
| topscore | 96.5 ± 0.2 | 83.2 ± 1.5 | 79.6 ± 1.2 | 84.9 ± 1.0 | 95.2 ± 0.5 | 87.2 ± 0.9 |
| resa2 | 99.1 ± 0.0 | 92.3 ± 0.1 | 90.2 ± 0.1 | 90.3 ± 0.2 | 97.5 ± 0.2 | 93.3 ± 0.1 |

Class sizes: Verb, seen 1814, Verb, unseen form 2585, Verb, unseen 3947, Nominal, hard 4173, Nominal, easy 4173.

## Top-1 errors by type (validation; share of all mentions)

| Model | Wrong top-1 | Time | Hierarchy | Other |
|---|---|---|---|---|
| Released | 57.4 | 1.8 | 11.6 | 44.0 |
| in_batch | 66.4 ± 1.9 | 1.9 ± 0.1 | 11.7 ± 0.6 | 52.8 ± 1.4 |
| uniform | 35.7 ± 0.2 | 2.1 ± 0.1 | 5.5 ± 0.2 | 28.1 ± 0.2 |
| topscore | 34.7 ± 1.2 | 2.1 ± 0.1 | 3.0 ± 0.1 | 29.6 ± 1.2 |
| resa2 | 30.5 ± 0.3 | 2.0 ± 0.1 | 4.4 ± 0.2 | 24.0 ± 0.1 |

## New York Times (mentions with a page, no Nil)

| Model | Top-1 | Recall@10 | Recall@100 |
|---|---|---|---|
| Released | 13.0 | 33.6 | 57.3 |  (n = 769)
| in_batch | 8.5 ± 1.1 | 24.4 ± 1.4 | 44.8 ± 1.7 |
| uniform | 9.8 ± 0.7 | 27.4 ± 1.0 | 50.8 ± 1.6 |
| topscore | 4.6 ± 0.7 | 14.4 ± 1.4 | 26.8 ± 1.8 |
| resa2 | 11.5 ± 0.1 | 30.9 ± 1.5 | 55.9 ± 0.4 |
