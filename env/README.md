# GPU environment

We run EVELINK on the newest torch (2.15.0, CUDA 13.2 wheels from PyPI),
not on the original torch 1.6. Reason: rented GPUs are Ampere or newer
(A40, A5000, L4, A100), and torch 1.6 does not support them.
See `docs/decisions.md` (2026-10-31).

On the GPU machine (Python 3.12 or newer):

```
python3 -m venv .venv && . .venv/bin/activate
pip install -r third_party/event-linking/requirements-torch2.txt
pip install -e .                      # our samplers (numpy only)
cd third_party/event-linking && python ../../scripts/smoke_evelink.py
```

The smoke test must print `OK` before the pilot starts.

The CUDA 13 wheels need an NVIDIA driver that supports CUDA 13
(`nvidia-smi` shows "CUDA Version: 13.x"). If the machine has an older
driver, install the CUDA 12 wheel from download.pytorch.org instead.

## What we changed in EVELINK for the new torch
All changes are in one commit after "Add EVELINK code, unchanged", so
`git diff` shows them. Nothing changes the math.
- `pytorch_transformers/optimization.py` (AdamW): the old call forms
  `add_(alpha, t)`, `addcmul_(v, a, b)`, `addcdiv_(v, a, b)` are removed in torch 2.
  We use the keyword forms `add_(t, alpha=...)` etc. Same update.
- `torch.load(...)`: torch 2.6+ loads only weights by default. We add
  `weights_only=False` for EVELINK's own files (candidate pools, encodings,
  checkpoints). Load only files from the EVELINK archive or our runs.
- `blink/biencoder/biencoder.py`, `blink/crossencoder/crossencoder.py`:
  `map_location` is now `"cpu"` (the old lambda gave a string, not a storage).
- `blink/biencoder/biencoder.py`: the in-batch target is `torch.arange(bs)`
  made on the device directly (old legacy constructor removed).
- `blink/main_dense.py`: imports flair (NER) only in interactive mode, so
  the old flair package is not needed for evaluation.

## Not checked yet
These changes are not run yet. The cloud session that made them was not
allowed to run EVELINK code. The smoke test above is the first check.
The pilot then compares recall with the paper's released bi-encoder.
