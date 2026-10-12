# resa2el

CS 2242 final project (Rodrigo Quezada). We compare negative samplers for the
retrieval stage of Event Linking (Yu et al., EACL 2023), on the Wikipedia data:
the original in-batch negatives, uniform, top-score, and ReSa2 (Li et al., ACML 2025).

## Questions
1. How does ReSa2 change the inclusion probability of each candidate, the
   coverage, and the variation between runs?
2. With a fixed compute budget, does ReSa2 put the correct page higher
   (recall at 10) than the original sampling?

## Folders
| Folder | What is in it |
|---|---|
| `src/resa2el/` | The samplers and the probability analysis (numpy, runs on CPU) |
| `tests/` | Tests for the samplers. A function that is not written yet is skipped. |
| `sim/` | Small model on fake scores (from the earlier reproduction work) |
| `lean/` | Lean 4 proofs. `cd lean && lake build` |
| `third_party/` | EVELINK and ReSa2 code (git submodules, added later) |
| `env/` | Environment for the GPU machine (EVELINK needs Python 3.8, torch 1.6) |
| `configs/` | One file per experiment: pilot and the 12 main runs |
| `scripts/` | Data download and run scripts |
| `docs/` | Decisions and the AI-usage log |

## Start (CPU work)
```
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Data
The data and the trained models are in one 35.6 GB archive. See `scripts/download_data.sh`.
Do not put data or checkpoints in git (`.gitignore` blocks `data/`, `checkpoints/`, `runs/`).
