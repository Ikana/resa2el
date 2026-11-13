# AI-usage log

| Date | Tool | What I asked | What I used | What I changed myself |
|---|---|---|---|---|
| 2026-10-12 | Claude | Set up the implementation repo | Folder layout, sampler stubs and tests, CI, configs | |
| 2026-11-02 | Claude | Move the GPU work from RunPod to AWS and run the EVELINK eval | AWS account, budget alerts, quota requests, prep and run scripts, `--cross_batch`, `--amp`, `--ckpt`, `--limit`, prefetch, `_modify_fast`, `scripts/check_cross_fast.py`, `docs/aws.md` | |
| 2026-11-04 | Claude | Write the samplers and the analysis helpers (Rodrigo asked) | `resa2` sampler, `simulate_inclusion`, `exact_stage1_inclusion`, one test that compares them | |
| 2026-11-04 | Claude | Training hook for the samplers (Rodrigo asked) | `scripts/build_pools.py`, `--sampler/--pools/--encoding/--freeze_cand_encoder/--max_updates` in `train_biencoder.py`, CPU smoke test | |
| 2026-11-06 | Claude | Question 1 analysis from the pools (Rodrigo asked) | `scripts/analyze_pools.py`, `docs/results/analysis/question1.md` | |
| 2026-11-06 | Claude | First paper draft from Rodrigo's narration, in the style of the Event Linking paper | `docs/paper/paper.tex` (two columns, related work, Lean section, limitations, appendices), `scripts/paper_figures.py` (3 figures) | |
| 2026-11-06 | Claude | Lean proof of the easy-negative bound L ≤ Σ e^(−g) (Rodrigo approved the paper item) | `lean/LossBound.lean` with Mathlib v4.34.1: `loss_eq_log_gaps`, `loss_le_sum_exp_neg_gap`, `loss_nonneg`, `loss_le_of_gap_ge` | |
| 2026-11-13 | Claude | Put the text that Rodrigo sent (made by an agent from the narration sheet) in sections 2, 3, 5.1, 6.4, limitations, and AI use | `docs/paper/paper.tex`, `docs/paper/paper.pdf` (16 pages); the AI section now says which sections are narration and which are not | |
| 2026-11-13 | Claude | Four new figures chosen by Rodrigo from a page of seven candidates (twenty draws for one mention, before and after, kinds of mistake, kinds of mention) | `scripts/paper_figures_extra.py`, `docs/paper/figures/*.pdf`, `docs/results/analysis/pools_sample200.npz`; the three tables of section 6.4 are replaced by figures (numbers stay in `docs/results/gaps/gaps.md`); Lean listings shortened; 16 pages | |
