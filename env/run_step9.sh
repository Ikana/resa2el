#!/bin/bash
# Two more seeds of the best new sampler, each followed by its 50% mix with the released model.
# Usage: run_step9.sh BACKSTOP_MINUTES NAME SAMPLER_ARGS...     Log: ~/step9.log
BACKSTOP=$1; NAME=$2; shift 2
sudo shutdown -h +$BACKSTOP
trap 'rc=$?; echo; echo STEP9_EXIT=$rc; date; sudo shutdown -h +2' EXIT
set -ex -o pipefail
E=~/resa2el/third_party/event-linking; P=$E/runs/plan
for S in 1 2; do
  echo "$NAME $S 1 $*" > ~/step9_runs.txt
  NO_SHUTDOWN=1 ~/run_plan.sh $BACKSTOP ~/step9_runs.txt
  ( . ~/venv/bin/activate; cd $E; export PYTHONPATH=.
    M=runs/plan/${NAME}_s$S/pytorch_model.bin
    if [ ! -f runs/plan/ranks/${NAME}mix50_s$S.npz ]; then
      python ../../scripts/mix_weights.py --start models/biencoder/pytorch_model.bin --models $M --alpha 0.50 --out runs/plan/mixed.bin
      python ../../scripts/eval_ranks.py --model runs/plan/mixed.bin --out runs/plan/ranks/${NAME}mix50_s$S.npz 2>&1 < /dev/null | grep -v "it/s]$"
      rm -f runs/plan/mixed.bin
    fi
    rm -f $M )
done
ls -la $P/ranks
echo STEP9_OK
