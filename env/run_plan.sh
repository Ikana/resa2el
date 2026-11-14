#!/bin/bash
# Runs of the results plan (docs/results/plan). One line of RUNS per run:
#   NAME SEED KEEP SAMPLER_ARGS...
# KEEP=1 keeps the trained model (needed for the test set), KEEP=0 deletes it after the evaluation.
# Skips runs that already have ranks, so a restart continues.
# Usage: run_plan.sh BACKSTOP_MINUTES RUNS_FILE      Log: ~/plan.log (start with tee).
BACKSTOP=${1:-300}
RUNS=${2:-~/plan_runs.txt}
sudo shutdown -h +$BACKSTOP
trap 'echo; echo PLAN_EXIT=$?; date; sudo shutdown -h +2' EXIT
set -ex -o pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
date
. ~/venv/bin/activate
cd ~/resa2el/third_party/event-linking
export PYTHONPATH=.
cat out/pools/all_entities_fp16.npy > /dev/null   # page vectors into the file cache (12 GB)
( while true; do echo "$(date +%T) $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader) ram_used_gb $(free -g | awk '/Mem/{print $3}')" >> ~/gpu_plan.log; sleep 60; done ) &
RUNDIR=runs/plan
mkdir -p $RUNDIR
COMMON="--data_path out --bert_model bert-large-uncased --out_dim 1 --pull_from_layer -1 --path_to_model models/biencoder/pytorch_model.bin --max_context_length 256 --max_cand_length 256 --train_batch_size 32 --gradient_accumulation_steps 4 --eval_batch_size 64 --num_train_epochs 1 --eval_interval 100000 --print_interval 50 --shuffle True --freeze_cand_encoder --pools out/pools/train.npz --encoding out/pools/all_entities_fp16.npy --max_updates ${UPDATES:-1500} --learning_rate 1e-5"
grep -v '^#' $RUNS | while read -r NAME SEED KEEP ARGS; do
  [ -z "$NAME" ] && continue
  OUT=$RUNDIR/${NAME}_s${SEED}
  if [ -f $RUNDIR/ranks/${NAME}_s${SEED}.npz ]; then echo "=== $OUT done, skip ==="; continue; fi
  echo "=== train $NAME seed $SEED: $ARGS ==="
  if [ ! -f $OUT/pytorch_model.bin ]; then
    rm -rf $OUT
    /usr/bin/time -v python blink/biencoder/train_biencoder.py --output_path $OUT --seed $SEED $COMMON $ARGS 2>&1 < /dev/null | grep -E "sampler |frozen|average loss|held-out|max_updates|Eval accuracy|training took|Elapsed|Maximum resident|Error|Traceback|^  File"
  fi
  rm -rf $OUT/epoch_0
  [ "${SKIP_EVAL:-0}" = 1 ] && continue
  echo "=== ranks of $NAME seed $SEED on valid and NYT ==="
  mkdir -p $RUNDIR/ranks
  python ../../scripts/eval_ranks.py --model $OUT/pytorch_model.bin --out $RUNDIR/ranks/${NAME}_s${SEED}.npz 2>&1 < /dev/null | grep -v "it/s]$"
  [ "$KEEP" = 1 ] || rm -f $OUT/pytorch_model.bin
  df -h / | tail -1
done
ls -la $RUNDIR/ranks
date
echo PLAN_OK
