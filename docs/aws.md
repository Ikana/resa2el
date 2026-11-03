# AWS setup for the EVELINK eval

RunPod is closed (see `runpod.md` for the old plan). The GPU work runs on AWS since 2026-11-02.

## Account
- AWS account **resa2** (ID 066080292532), a member of Rodrigo's AWS Organization. It holds only this project.
- CLI profile on the Mac: `AWS_PROFILE=resa2` (it assumes `OrganizationAccountAccessRole` from the default profile).
- Region: us-east-1. Budget alert `resa2-total`: email at 25%, 50% and 90% of $100.
- Quotas (approved 2026-11-02): 64 vCPU standard, 8 vCPU G on-demand, 8 vCPU G spot.

## Instance
- One instance, `resa2-eval` (`i-0e87d9c7cc7ceb86b`), with one 200 GB gp3 root disk. The disk keeps the data, the models, the venv and the outputs when the instance is stopped. A stopped instance costs only the disk (about $0.50 per day).
- Change the instance type for each phase (the instance must be stopped):
  - CPU work (download, `data_processing.py`, `--build_lowmem`): `r6i.2xlarge` (64 GB RAM, $0.50/h).
  - GPU work (predict): `g5.2xlarge` (A10G 24 GB, 32 GB RAM, $1.21/h on demand).
  - Read logs or copy results: `t3.large` ($0.08/h).
- SSH: `ssh -i ~/.ssh/resa2.pem ubuntu@<public ip>`. The security group allows port 22 from the Mac's IP only. When the Mac's IP changes, add the new `/32`.
- Paths on the instance: data and models in `/opt/resa2/dl/unz`, symlinked as `data/` and `models/` in `~/resa2el/third_party/event-linking`; Python venv in `~/venv`; low-memory store in `out/lowmem` (33 GB).

Commands (from the Mac):
```
AWS_PROFILE=resa2 aws ec2 modify-instance-attribute --instance-id i-0e87d9c7cc7ceb86b --instance-type '{"Value":"g5.2xlarge"}'
AWS_PROFILE=resa2 aws ec2 start-instances --instance-ids i-0e87d9c7cc7ceb86b
AWS_PROFILE=resa2 aws ec2 stop-instances --instance-ids i-0e87d9c7cc7ceb86b
```

## Rules
- Every run script sets a backstop: `sudo shutdown -h +MINUTES` at the start, and `sudo shutdown -h +2` in an `EXIT` trap. A stopped instance does not cost GPU money.
- Write each GPU session in `compute-log.md`.
- Pipe the run output through `tee`, not `tail`: when the backstop kills the job, `tail` loses everything.

## Fast eval options (added 2026-11-03)
The released crossencoder is BERT-large. In float32 it needs about 10 GPU hours on an A10G for the Wikipedia test set (19,267 mentions x 100 candidates x 511 tokens). Options in `event_linking_main.py`:
- `--cross_batch N`: crossencoder batch size in mentions (N x 100 sequences per forward). Batch 16 (the default from `params.json`) does not fit in 24 GB.
- `--amp fp16|bf16|tf32`: mixed precision for the crossencoder only. Attention runs fused (`scaled_dot_product_attention`) and the final score in float32, so only rounding changes. The bi-encoder stays in float32.
- `--ckpt DIR`: saves the bi-encoder top-k and each crossencoder chunk in DIR. A stopped run continues from there with the same arguments.
- `--limit N`: the first N mentions only, for timing tests. The output files get the suffix `_limitN`.
- Prefetch (on by default): the next chunk is tokenized on the CPU while the GPU scores the current one. `--no_prefetch` turns it off.
- `python ../../scripts/check_cross_fast.py [--real]` checks these paths on the CPU. It prints `CROSS_FAST_CHECK=PASS`.

## Costs so far
See `compute-log.md`. RunPod: about $50 before the move. AWS: about $12 up to 2026-11-03 morning.
