#!/bin/bash
# Step 7 of the results plan: test set, once. The list is fixed in docs/results/plan/test_list.md.
# Usage: run_test.sh BACKSTOP_MINUTES     Log: ~/test.log
BACKSTOP=${1:-300}
sudo shutdown -h +$BACKSTOP
trap 'rc=$?; echo; echo TEST_EXIT=$rc; date; [ "${NO_SHUTDOWN:-0}" = 1 ] || sudo shutdown -h +2' EXIT
set -ex -o pipefail
. ~/venv/bin/activate
cd ~/resa2el/third_party/event-linking
export PYTHONPATH=.
R=runs/main_lr1e-5; P=runs/plan; START=models/biencoder/pytorch_model.bin; OUT=runs/plan/ranks_test; TMP=runs/plan/mixed.bin
mkdir -p $OUT
one() {  # NAME MODEL
  [ -f $OUT/$1.npz ] && { echo "=== $1 done, skip ==="; return; }
  echo "=== test $1 ==="
  python ../../scripts/eval_ranks.py --test --model $2 --out $OUT/$1.npz 2>&1 < /dev/null | grep -v "it/s]$"
}
one released $START
for S in 0 1 2; do one resa2_s$S $R/resa2_s$S/pytorch_model.bin; done
for S in 0 1 2; do
  [ -f $OUT/mix50_s$S.npz ] && continue
  python ../../scripts/mix_weights.py --start $START --models $R/resa2_s$S/pytorch_model.bin --alpha 0.50 --out $TMP
  one mix50_s$S $TMP
  rm -f $TMP
done
for S in 0 1 2; do one uniform_s$S $R/uniform_s$S/pytorch_model.bin; done
for S in 0 1 2; do one stage2only_s$S $P/stage2only_s$S/pytorch_model.bin; done
one allpool_s0 $P/allpool_s0/pytorch_model.bin
ls -la $OUT
echo TEST_OK
