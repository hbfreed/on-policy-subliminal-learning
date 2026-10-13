#!/bin/bash
# Train a student in segments ending at each epoch of SCHEDULE; after each segment (adapter +
# resumable state saved), evaluate that epoch on EVAL_GPU, log the table, then resume.
#   student_supervisor.sh OUT STEPS_PER_EPOCH "1 2 3 4 6 8 ..." -- <train_student_opd.py args>
# (the train args must not include --out / --max-steps; this script adds them)
# Env: TRAIN_GPUS (physical cards for the training process, e.g. "2,3": inside, cuda:0 is the
# first (vLLM for on-policy) and cuda:1 the next; default all), EVAL_GPU (physical card for the
# eval pauses; default the first of TRAIN_GPUS, else 0).
cd "$(dirname "$0")/.." || exit 1
OUT=$1; SPE=$2; SCHEDULE=$3; shift 4
EVAL_GPU=${EVAL_GPU:-${TRAIN_GPUS%%,*}}; EVAL_GPU=${EVAL_GPU:-0}
VIS=(); [ -n "$TRAIN_GPUS" ] && VIS=(env CUDA_VISIBLE_DEVICES=$TRAIN_GPUS)
NAME=$(basename $OUT); S=runs/pipeline.status; EV=${OUT}_eval
say(){ echo "$(date +%H:%M) [$NAME] $*" >> $S; }
mkdir -p $EV; [ -d $EV/base ] || [ ! -d runs/teacher_sft_s1_eval/base ] || cp -r runs/teacher_sft_s1_eval/base $EV/

for E in $SCHEDULE; do
  if [ ! -d $OUT/ep$E ]; then
    say "training to epoch $E (step $((E * SPE)))"
    # own process group, so a stop kills python and any vLLM engine together
    setsid --wait "${VIS[@]}" .venv-vllm/bin/python on_policy/train_student_opd.py "$@" --out $OUT \
      --max-steps $((E * SPE)) >> $OUT.log 2>&1
    rc=$?
    [ $rc = 0 ] && [ -d $OUT/ep$E ] || { say "FAIL training segment to epoch $E (exit $rc)"; exit 1; }
  fi
  if [ ! -f $EV/ep$E/summary.json ]; then
    scripts/eval_retry.sh $EVAL_GPU $EV $OUT/ep$E
    say "eval ep$E: $(.venv-vllm/bin/python scripts/table.py $EV | grep "^ep$E " | tr -s " ")"
  fi
done
say "DONE schedule ($SCHEDULE)"
