# Handoff prompt: improve the ReSa2 paper

Paste everything below the line into the agent. Written on 2026-11-27.

---

You take over a class project from another AI agent. Your job is to try to make the paper better. Work in a new branch so the two versions can be compared.

## The project

- Repository: `Ikana/resa2el` on GitHub. Clone it with `gh repo clone Ikana/resa2el`.
- Start from the branch `claude/project-thread-jn1e5q`. Create your own branch from it, for example `neon/paper-attempt`. Do not push to `main` and do not push to the Claude branch.
- The paper is `docs/paper/paper.tex`. Build it with `pdflatex` three times in `docs/paper`. The PDF is `docs/paper/paper.pdf`.
- Course: COMPSCI 2242, a class on probabilistic analysis and algorithms. The author is Rodrigo. The paper is due on December 10, 2026.
- The paper is an implementation paper. It takes EVELINK (Yu et al., 2023, event linking to Wikipedia) and changes the negative sampler in the training of the first stage, the bi-encoder retriever.

Read these first, in this order:

1. `AGENTS.md` (rules for AI agents in this repository)
2. `docs/paper/paper.tex`
3. `docs/results/plan/` (every result table, with logs): `steps1_5.md`, `steps2_3.md`, `steps4_6.md`, `step7_test.md`, `step8.md`, `test2.md`, `push_mass.json`
4. `docs/results/gaps/gaps.md`
5. `docs/ai-usage-log.md` (what the earlier agent did, session by session)
6. `docs/aws.md` and `docs/compute-log.md`

## Hard rules

- The paper is at most 16 pages, single column, appendix included. It is at 16 now.
- Write in Simplified Technical English: short sentences, simple words. Write "I", not "we". The author must be able to explain every sentence.
- The instructor asked for three things: the task description up front, more math, and no account of paths that were not taken.
- The author chose the structure: Part I (ReSa2) and an Epilogue (what came after ReSa2). Each part gives the math first and the experiments second. Keep that unless Rodrigo tells you to change it.
- Do not mention the bandit experiment in the paper. Rodrigo asked for that.
- Every proof goes in `lean/` and must pass `lake build` (Lean 4 v4.34.1 with Mathlib). Say in the paper which statements Lean checks and which are proved by hand.
- **Do not evaluate any model on the Wikipedia test set.** It was used in two rounds, each with a list fixed in writing first (`test_list.md`, `test_list2.md`). A third round would make the test numbers worthless. Use validation data for all new work.
- Do not invent numbers. Every number in the paper must come from a file in `docs/results/` or from a run that you made and logged.
- Add a row to `docs/ai-usage-log.md` for each work session. Name your tool and model.
- Ask Rodrigo before you spend money. Ask one question at a time.
- Do not submit, email or publish anything.

## Where the work stands

Setup of every run: start from the released EVELINK bi-encoder, page encoder frozen, 1,500 updates, batch 32, learning rate 1e-5, 7 negatives for each mention from a pool (the released model's top 1,000 pages). Evaluation on validation (16,692 mentions) and on a New York Times set (769 mentions).

| Sampler (7 negatives unless noted) | Valid top-1 | Valid R@10 | News R@10 |
|---|---|---|---|
| Released model | 42.6 | 77.7 | 33.6 |
| In-batch | 33.6 | 69.6 | 24.4 |
| Uniform from the pool | 64.3 | 92.2 | 27.4 |
| Top-score, fixed scores | 65.3 | 87.2 | 14.4 |
| ReSa2 | 69.5 | 93.3 | 30.9 |
| Live hard negatives (top 7 by current score) | 75.6 | 94.7 | 30.0 |
| All 999 pool pages | 76.0 | 94.7 | 31.7 |
| Live hard negatives, weights averaged 50% with released | 67.6 | 92.4 | 36.3 |

The math in the paper: the negatives enter the loss and the size of the update only through one sum, their "mass". The top k pages by current score have the most mass that k pages can have. Seven uniform pages hold k/N of the mass in expectation. A condition for when the 50% weight average beats both models. Lean files: `LossBound.lean`, `Sampling.lean`, `Mix.lean`, `FalseNegatives.lean`.

## Known weak points (good places to improve)

1. **The explanation for top-score is not measured.** The paper says that fixed top-score fails because the mass moves to other pages during training. The mass was measured only before the first update, on 200 pools. A measurement of the mass during training would turn an explanation into a result.
2. **Rung 1 has a bound and no measurement.** The mass of in-batch negatives is given as "at most 0.7%".
3. **The news result has one data set.** There are 769 news mentions, they were looked at many times, and the choice of sampler used them. Nothing confirms the news gain of the weight average.
4. **Wrong-year mistakes did not fall.** This was the original motivation. No sampler reduced them (about 2% of validation mentions before and after). A sampler or a loss that targets pages with the same title and another year was never tried.
5. **The curvature constant in the proposition on the weight average is not measured**, and the "parabola" condition was checked against accuracy at five points only.
6. **Single runs.** Some rows have one run (all pool pages, 63 negatives, the similarity bands, three of the four rules with current scores).
7. **The page encoder is frozen and the pool is fixed.** Results can differ when both encoders train or the pool is refreshed.
8. **Writing.** Check that the abstract, introduction and conclusion do not repeat each other, and that every claim in the text matches its table.

You do not have to take these. First read the paper as a strict reviewer and write your own list of the five most important problems. Show that list to Rodrigo before you change anything.

## How to use AWS

Ask Rodrigo for a budget number before the first GPU start. The earlier budget is almost used up.

**Access**

- AWS CLI profile: `resa2`. Region: `us-east-1`. Use `export AWS_PROFILE=resa2 AWS_DEFAULT_REGION=us-east-1`.
- Instance: `i-0e87d9c7cc7ceb86b`, in `us-east-1b`. It is **stopped** now, with the type `t3.large`.
- Disk: `vol-07930d44ba40c9edb`, 200 GB gp3. It costs about $0.50 per day while it exists, also when the instance is stopped. It holds all data and models. Do not delete it.
- SSH: user `ubuntu`, key `~/.ssh/resa2.pem`. The public IP changes at each start.
- Security group: `sg-0b247f861e82ce99e`. It allows port 22 from single IP addresses only.

**Start a GPU session**

```bash
export AWS_PROFILE=resa2 AWS_DEFAULT_REGION=us-east-1
I=i-0e87d9c7cc7ceb86b
# 1. Set the GPU type while the instance is stopped. g5.2xlarge = one A10G 24 GB, about $1.21 per hour.
aws ec2 modify-instance-attribute --instance-id $I --instance-type Value=g5.2xlarge
# 2. Start. "InsufficientInstanceCapacity" is common. Wait 20 seconds and try again, up to 10 times.
aws ec2 start-instances --instance-ids $I
aws ec2 wait instance-running --instance-ids $I
# 3. Get the new IP.
IP=$(aws ec2 describe-instances --instance-ids $I --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
# 4. Allow this computer. An error that says "Duplicate" is fine.
aws ec2 authorize-security-group-ingress --group-id sg-0b247f861e82ce99e --protocol tcp --port 22 --cidr $(curl -s https://checkip.amazonaws.com)/32
# 5. Connect.
ssh -i ~/.ssh/resa2.pem ubuntu@$IP
```

**On the instance**

- Code and data: `~/resa2el/third_party/event-linking/`. Python environment: `. ~/venv/bin/activate`.
- This tree is **not a git repository**. Copy changed files to it with `scp`. Copy results back with `scp` and commit them on your computer.
- Released model: `models/biencoder/pytorch_model.bin`. Stored page vectors: `out/pools/all_entities_fp16.npy` (12 GB). Pools for training: `out/pools/train.npz`.
- Trained models that still exist: `runs/main_lr1e-5/<sampler>_s<seed>/pytorch_model.bin` for in_batch, uniform, topscore and resa2 with seeds 0 to 2, and `runs/plan/allpool_s0`, `runs/plan/stage2only_s0..2`. The live hard negatives models were deleted to save disk. To get one back, train it again (same seed gives the same validation result).
- Rank files of every evaluation: `runs/plan/ranks/*.npz`, `runs/test2/ranks/*.npz`.
- **The disk is 96% full: about 8.5 GB free.** One saved model is 2.7 GB. Keep at most two new models at a time. Evaluate, copy the rank file, delete the model.

**Run a training**

The script `~/run_plan.sh` (in the repository: `env/run_plan.sh`) trains and evaluates a list of runs. One line of the list is `NAME SEED KEEP SAMPLER_ARGS`.

```bash
echo "myrun 0 0 --sampler live_top --negatives 7" > ~/my_runs.txt
RUNDIR=runs/neon nohup ~/run_plan.sh 300 ~/my_runs.txt > ~/neon.log 2>&1 < /dev/null &
```

- The first argument (300) is a safety limit in minutes. The machine shuts itself down after that time, and also 2 minutes after the script ends. Set `NO_SHUTDOWN=1` if you chain several scripts, and then shut down yourself.
- `KEEP=0` deletes the model after the evaluation. `KEEP=1` keeps it.
- Samplers that exist: `in_batch`, `uniform`, `topscore`, `resa2`, `stage2_only`, `band`, `allpool`, `gold_random`, `mixed`, `live_top`, `live_top_below`, `score_prop`. The code is in `src/resa2el/samplers.py`, `src/resa2el/candidates.py` and `third_party/event-linking/blink/biencoder/train_biencoder.py`. Tests: `PYTHONPATH=src python -m pytest -q tests` (44 pass).
- One run with its evaluation takes about 55 to 62 minutes, about $1.20. One evaluation alone takes about 13 minutes.
- Evaluation script: `scripts/eval_ranks.py --model PATH --out FILE.npz` gives validation and news ranks. **Never pass `--test`.**
- Weight average: `scripts/mix_weights.py --start models/biencoder/pytorch_model.bin --models PATH --alpha 0.5 --out mixed.bin`.

**Stop. This step is not optional.**

```bash
aws ec2 stop-instances --instance-ids $I
aws ec2 wait instance-stopped --instance-ids $I
aws ec2 modify-instance-attribute --instance-id $I --instance-type Value=t3.large
aws ec2 describe-instances --instance-ids $I --query 'Reservations[0].Instances[0].State.Name' --output text
```

- A running g5.2xlarge costs about $29 per day. Check that the state is `stopped` at the end of every session, and tell Rodrigo the state and the cost of the session.
- To copy files without a GPU, start the instance as `t3.large` (about $0.08 per hour) and stop it after.
- Do not create new instances, volumes, snapshots or buckets without asking Rodrigo.
- Write each session in `docs/compute-log.md`: date, hours, type, cost, what ran.

## What to deliver

1. Your list of the five most important problems of the paper, for Rodrigo, before any change.
2. After his answer: the changes on your branch, with the PDF at 16 pages or less, `lake build` passing, the tests passing, and a short list of what you changed and why.
3. For every new number: the file in `docs/results/` that holds it and the log of the run.
