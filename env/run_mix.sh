#!/bin/bash
# Step 1 of the results plan: weight mixing, evaluation only.
# 3 ReSa2 seeds x alpha 0.25, 0.50, 0.75 mixed with the released model, plus the average of the 3 seeds.
# Usage: run_mix.sh BACKSTOP_MINUTES     Log: ~/mix.log
BACKSTOP=${1:-240}
sudo shutdown -h +$BACKSTOP
trap 'rc=$?; echo; echo MIX_EXIT=$rc; date; [ "${NO_SHUTDOWN:-0}" = 1 ] || sudo shutdown -h +2' EXIT
set -ex -o pipefail
. ~/venv/bin/activate
cd ~/resa2el/third_party/event-linking
export PYTHONPATH=.
R=runs/main_lr1e-5; START=models/biencoder/pytorch_model.bin; OUT=runs/plan/ranks; TMP=runs/plan/mixed.bin
one() {  # NAME ALPHA MODELS...
  NAME=$1; ALPHA=$2; shift 2
  [ -f $OUT/$NAME.npz ] && { echo "=== $NAME done, skip ==="; return; }
  echo "=== mix $NAME ==="
  python ../../scripts/mix_weights.py --start $START --models "$@" --alpha $ALPHA --out $TMP
  python ../../scripts/eval_ranks.py --model $TMP --out $OUT/$NAME.npz 2>&1 < /dev/null | grep -v "it/s]$"
  rm -f $TMP
}
for A in 50 25 75; do for S in 0 1 2; do one mix${A}_s$S 0.$A $R/resa2_s$S/pytorch_model.bin; done; done
one soup_s012 1.0 $R/resa2_s0/pytorch_model.bin $R/resa2_s1/pytorch_model.bin $R/resa2_s2/pytorch_model.bin
ls -la $OUT
echo MIX_OK
