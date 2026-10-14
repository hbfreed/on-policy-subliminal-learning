#!/bin/bash
# usage: eval_retry.sh GPU OUTDIR ADAPTER... ; one adapter per vLLM process, up to 3 tries,
# escalating around vLLM 0.31's intermittent crashes in LoRA prompt-logprob scoring:
# try 1 as is, try 2 without CUDA graphs, try 3 with the margins scored by HF instead.
cd "$(dirname "$0")/.." || exit 1
g=$1; out=$2; shift 2
for a in "$@"; do
  n=$(basename $a)
  for try in 1 2 3; do
    [ -f $out/$n/summary.json ] && break
    case $try in 1) extra="";; 2) extra="--eager";; 3) extra="--eager --hf-margin";; esac
    CUDA_VISIBLE_DEVICES=$g timeout 1500 .venv-vllm/bin/python on_policy/eval_vllm.py --adapters $a --out $out $extra >> $out.$n.log 2>&1
    echo "$a try $try ($extra) exit $?" >> $out.retry.log
  done
done
echo DONE >> $out.retry.log
