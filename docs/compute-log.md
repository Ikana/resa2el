# Compute log

One line for each GPU session.

| Date | Pod (GPU) | Hours | Cost | What ran |
|---|---|---|---|---|
| 2026-11-02 | AWS r6i.2xlarge (no GPU, 64 GB RAM) | 0.8 | $0.45 | Download (34 GB), data_processing.py, --build_lowmem. LOWMEM_CHECK=PASS. |
| 2026-11-03 | AWS g5.2xlarge (A10G 24 GB) | 0.4 | $0.50 | Test predict, crossencoder batch 16: CUDA out of memory. |
| 2026-11-03 | AWS g5.2xlarge (A10G 24 GB) | 8.0 | $9.70 | Test predict, --cross_batch 1, float32: the 8 h timer stopped it before the end. No result. |
| 2026-11-03 | AWS g5.2xlarge (A10G 24 GB) | 0.65 | $0.80 | Timing pilot, 400 test mentions: float32 2.97 s/mention; fp16 + fused attention 0.94 s/mention, same top-1 on 400/400. |
| 2026-11-03 | AWS g5.2xlarge (A10G 24 GB) | 6.0 | $7.30 | Full eval, fp16 + fused attention, batch 4: Wikipedia test 5 h 25 min, NYT 36 min. Results in `docs/results/`. |
| 2026-11-04 | AWS g5.2xlarge (A10G 24 GB) | 2.0 | $2.40 | Pools for valid and train (correct page in its pool for 98%); released recall on valid: top-1 42.6, R@10 77.7, R@100 93.1. Training crashed in EVELINK's validation (last batch smaller than eval_batch_size). |
| 2026-11-04 | AWS g5.2xlarge (A10G 24 GB) | 1.4 | $1.70 | Pilot try 2: tokenization (27 min, now cached), then CUDA out of memory at batch 32 in one step. |
| 2026-11-04 | AWS g5.2xlarge (A10G 24 GB) | 2.3 | $2.80 | Pilot try 3: uniform and resa2, 1,500 updates each (batch 32 = 4 x 8), 45 min per run + 11 min recall eval; 13 GB GPU. Valid R@10: released 77.7, uniform 90.0, resa2 92.0. Results in `docs/results/pilot/`. |
| 2026-11-04/05 | AWS g5.2xlarge (A10G 24 GB) | 11.0 | $13.30 | 12 main runs (4 samplers x 3 seeds, 1,500 updates). resa2 92.0 R@10 on healthy seeds vs uniform 90.3; topscore s1 and resa2 s1 collapsed. Plus 0.7 h lost to a disk-full crash. Results in `docs/results/main/`. |
| 2026-11-05 | AWS g5.2xlarge (A10G 24 GB) | 2.1 | $2.50 | Recovery test: the collapsed topscore s1 model + 1,500 updates with resa2 and with uniform negatives. No recovery (R@10 0.0). |
| 2026-11-05/06 | AWS g5.2xlarge (A10G 24 GB) | 12.0 | $14.50 | 12 main runs at lr 1e-5. No collapse. resa2 93.3 ± 0.1 R@10, uniform 92.2, topscore 87.2, in_batch 69.6. Results in `docs/results/main_lr1e-5/`. |
