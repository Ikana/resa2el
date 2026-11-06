# Question 1: inclusion probability, coverage, and variation between runs

Validation pools, 500 random mentions, 200 runs of each sampler per mention, k = 7 negatives.

| Sampler | expected negatives per run by similarity rank to the correct page: top 25 / 25-75 / 75-300 / 300-1000 | distinct pages selected (of 1,000) | overlap of two runs (Jaccard) | mean score gap to the correct page | mean similarity rank |
|---|---|---|---|---|---|
| uniform | 0.17 / 0.34 / 1.55 / 4.95 | 756 | 0.00 | -13.3 | 504 |
| topscore | 4.13 / 1.07 / 1.17 / 0.62 | 7 | 1.00 | -1.4 | 83 |
| resa2 | 2.15 / 3.98 / 0.87 / 0.00 | 84 | 0.05 | -8.6 | 43 |

How to read it:
- A row sums to 7 (the 7 negatives of one run), split by the negative's similarity rank to the correct page (rank 0 = the most similar page in the pool, often a 'wrong year' look-alike).
- Coverage: how many different pool pages a sampler ever selects over 200 runs (uniform: close to all; topscore: exactly 7).
- Jaccard: 1.0 means two runs select the same 7 pages (no randomness); 0 means no overlap.
- Score gap: negative means the negatives score below the correct page (easy); positive means above (hard).
