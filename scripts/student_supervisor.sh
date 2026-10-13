#!/bin/bash
# Train a student in segments ending at each epoch of SCHEDULE; after each segment (adapter +
# resumable state saved), evaluate that epoch on GPU 0, log the table, then resume.
#   student_supervisor.sh OUT STEPS_PER_EPOCH "1 2 3 4 6 8 ..." -- <train_student_opd.py args>
# (the train args must not include --out / --max-steps; this script adds them)
cd "$(dirname "$0")/.." || exit 1
OUT=$1; SPE=$2; SCHEDULE=$3; shift 4
NAME=$(basename $OUT); S=runs/pipeline.status; EV=${OUT}_eval
say(){ echo "$(date +%H:%M) [$NAME] $*" >> $S; }
mkdir -p $EV; [ -d $EV/base ] || [ ! -d runs/teacher_sft_s1_eval/base ] || cp -r runs/teacher_sft_s1_eval/base $EV/

for E in $SCHEDULE; do
  if [ ! -d $OUT/ep$E ]; then
    say "training to epoch $E (step $((E * SPE)))"
    # own process group, so a stop kills python and any vLLM engine together
    setsid --wait .venv-vllm/bin/python on_policy/train_student_opd.py "$@" --out $OUT \
      --max-steps $((E * SPE)) >> $OUT.log 2>&1
    rc=$?
    [ $rc = 0 ] && [ -d $OUT/ep$E ] || { say "FAIL training segment to epoch $E (exit $rc)"; exit 1; }
  fi
  if [ ! -f $EV/ep$E/summary.json ]; then
    scripts/eval_retry.sh 0 $EV $OUT/ep$E
    say "eval ep$E: $(.venv-vllm/bin/python scripts/table.py $EV | grep "^ep$E " | tr -s " ")"
  fi
done
say "DONE schedule ($SCHEDULE)"
