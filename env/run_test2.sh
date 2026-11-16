#!/bin/bash
# Second test round. The list is fixed in docs/results/plan/test_list2.md.
# Trains live_top again (seeds 0, 1, 2), evaluates on validation and NYT, then one time on the test set.
# Seeds 1 and 2: also the 50% mix with the released model on the test set. One model on disk at a time.
# Usage: run_test2.sh BACKSTOP_MINUTES     Log: ~/test2.log
BACKSTOP=${1:-360}
sudo shutdown -h +$BACKSTOP
trap 'rc=$?; echo; echo TEST2_EXIT=$rc; date; sudo shutdown -h +2' EXIT
set -ex -o pipefail
E=~/resa2el/third_party/event-linking; D=runs/test2; OUT=$D/ranks_test; START=models/biencoder/pytorch_model.bin
one() {  # NAME MODEL
  [ -f $OUT/$1.npz ] && { echo "=== $1 done, skip ==="; return; }
  echo "=== test $1 ==="
  python ../../scripts/eval_ranks.py --test --model $2 --out $OUT/$1.npz 2>&1 < /dev/null | grep -v "it/s]$"
}
for S in 0 1 2; do
  echo "livetop $S 1 --sampler live_top --negatives 7" > ~/test2_runs.txt
  RUNDIR=$D NO_SHUTDOWN=1 ~/run_plan.sh $BACKSTOP ~/test2_runs.txt
  ( . ~/venv/bin/activate; cd $E; export PYTHONPATH=.; mkdir -p $OUT
    M=$D/livetop_s$S/pytorch_model.bin
    if [ -f $M ]; then
      one livetop_s$S $M
      if [ $S != 0 ] && [ ! -f $OUT/livetopmix50_s$S.npz ]; then
        python ../../scripts/mix_weights.py --start $START --models $M --alpha 0.50 --out $D/mixed.bin
        one livetopmix50_s$S $D/mixed.bin
        rm -f $D/mixed.bin
      fi
      rm -f $M
    fi
    df -h / | tail -1 )
done
ls -la $E/$OUT $E/$D/ranks
echo TEST2_OK
