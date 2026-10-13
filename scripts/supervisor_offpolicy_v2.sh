#!/bin/bash
# Unattended: build the 12 shards on 3 GPUs from a shared queue (fresh log per attempt, kill
# on vLLM illegal-memory crash, up to 6 attempts per shard), merge, verify vs the paper's
# loss, then launch the off-policy student arm. Progress -> runs/pipeline.status
cd "$(dirname "$0")/.." || exit 1
O=runs/offpolicy_sft_s1; S=runs/pipeline.status; T=runs/teacher_sft_s1/ep7
[ -d $O ] || exit 1
say(){ echo "$(date +%H:%M) $*" >> $S; }

attempt(){  # gpu shard try -> prints exit code
  local g=$1 k=$2 n=$3 log=$O/s$2.try$3.log
  CUDA_VISIBLE_DEVICES=$g timeout 7200 .venv-vllm/bin/python on_policy/build_offpolicy_data.py \
    --teacher $T --shard $k --num-shards 12 --max-num-seqs 112 --out $O > $log 2>&1 &
  local pid=$!
  while kill -0 $pid 2>/dev/null; do
    if grep -q "illegal memory access" $log; then
      say "shard $k try $n: vLLM illegal memory access, killing"
      # timeout leads its own process group (python + the vLLM EngineCore): kill it whole,
      # or the engine is orphaned holding the GPU
      kill $pid 2>/dev/null; sleep 10; kill -9 -- -$pid 2>/dev/null
    fi
    sleep 30
  done
  wait $pid
  echo $?
}

worker(){  # gpu
  local g=$1 k i n rc
  while true; do
    k=""
    for i in $(seq 0 11); do
      [ -f $O/s$i.jsonl ] && continue
      [ $(ls $O/s$i.try*.log 2>/dev/null | wc -l) -ge 6 ] && continue
      mkdir $O/claim_$i 2>/dev/null && { k=$i; break; }
    done
    [ -z "$k" ] && return
    n=$(( $(ls $O/s$k.try*.log 2>/dev/null | wc -l) + 1 ))
    rc=$(attempt $g $k $n)
    say "shard $k try $n on gpu $g exit $rc"
    rmdir $O/claim_$k
  done
}

rmdir $O/claim_* 2>/dev/null
say "supervisor start: $(ls $O/s*.jsonl 2>/dev/null | wc -l)/12 shards done"
for g in 0 1 2; do worker $g & sleep 2; done
wait
n=$(ls $O/s*.jsonl | wc -l)
[ $n -lt 12 ] && { say "FAIL build: only $n/12 shards after 6 attempts each"; exit 1; }
say "shards done; merging"
.venv-vllm/bin/python on_policy/build_offpolicy_data.py --merge --out $O > $O/merge.log 2>&1 || { say "FAIL merge"; exit 1; }
say "merged: $(tail -1 $O/merge.log)"
CUDA_VISIBLE_DEVICES=0 .venv-vllm/bin/python on_policy/verify_offpolicy_loss.py --teacher $T --data $O/train.jsonl > $O/verify.log 2>&1
rc=$?
say "verify: $(grep "^\[verify\]" $O/verify.log | tr "\n" " ")"
[ $rc = 0 ] || { say "FAIL verify, not launching"; exit 1; }
say "LAUNCH student_offpolicy_s1"
.venv-vllm/bin/python on_policy/train_student_opd.py --teacher $T --rollout dataset --data $O/train.jsonl \
  --out runs/student_offpolicy_s1 --run-name student_offpolicy_s1 > runs/student_offpolicy_s1.log 2>&1
say "student_offpolicy_s1 EXIT $?"
