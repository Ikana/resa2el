# Compute log

One line for each GPU session.

| Date | Pod (GPU) | Hours | Cost | What ran |
|---|---|---|---|---|
| 2026-11-02 | AWS r6i.2xlarge (no GPU, 64 GB RAM) | 0.8 | $0.45 | Download (34 GB), data_processing.py, --build_lowmem. LOWMEM_CHECK=PASS. |
| 2026-11-03 | AWS g5.2xlarge (A10G 24 GB) | 0.4 | $0.50 | Test predict, crossencoder batch 16: CUDA out of memory. |
| 2026-11-03 | AWS g5.2xlarge (A10G 24 GB) | 8.0 | $9.70 | Test predict, --cross_batch 1, float32: the 8 h timer stopped it before the end. No result. |
