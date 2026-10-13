#!/bin/bash
# usage: eval_retry.sh GPU OUTDIR ADAPTER... ; one adapter per vLLM process, up to 3 tries
cd "$(dirname "$0")/.." || exit 1
g=$1; out=$2; shift 2
for a in "$@"; do
  n=$(basename $a)
  for try in 1 2 3; do
    [ -f $out/$n/summary.json ] && break
    CUDA_VISIBLE_DEVICES=$g timeout 1200 .venv-vllm/bin/python on_policy/eval_vllm.py --adapters $a --out $out >> $out.$n.log 2>&1
    echo "$a try $try exit $?" >> $out.retry.log
  done
done
echo DONE >> $out.retry.log
