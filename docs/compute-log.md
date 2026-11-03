# Compute log

One line for each GPU session.

| Date | Pod (GPU) | Hours | Cost | What ran |
|---|---|---|---|---|
| 2026-11-02 | AWS r6i.2xlarge (no GPU, 64 GB RAM) | 0.8 | $0.45 | Download (34 GB), data_processing.py, --build_lowmem. LOWMEM_CHECK=PASS. |
| 2026-11-03 | AWS g5.2xlarge (A10G 24 GB) | 0.4 | $0.50 | Test predict, crossencoder batch 16: CUDA out of memory. |
| 2026-11-03 | AWS g5.2xlarge (A10G 24 GB) | 8.0 | $9.70 | Test predict, --cross_batch 1, float32: the 8 h timer stopped it before the end. No result. |
| 2026-11-03 | AWS g5.2xlarge (A10G 24 GB) | 0.65 | $0.80 | Timing pilot, 400 test mentions: float32 2.97 s/mention; fp16 + fused attention 0.94 s/mention, same top-1 on 400/400. |
| 2026-11-03 | AWS g5.2xlarge (A10G 24 GB) | 6.0 | $7.30 | Full eval, fp16 + fused attention, batch 4: Wikipedia test 5 h 25 min, NYT 36 min. Results in `docs/results/`. |
