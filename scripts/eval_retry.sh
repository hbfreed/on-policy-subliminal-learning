#!/bin/bash
# usage: eval_retry.sh GPU OUTDIR ADAPTER... ; one adapter per vLLM process, up to 3 tries,
# escalating around vLLM 0.31's intermittent crashes in LoRA prompt-logprob scoring:
# margins are scored with HF (as the paper's eval_local.py) and vLLM only generates; the
# last try also disables CUDA graphs. Env EVAL_ARGS is appended (e.g. "--hf-bs 4" on 24GB cards).
cd "$(dirname "$0")/.." || exit 1
g=$1; out=$2; shift 2
# HF margin scoring OOMs at its default batch on 24 GB cards: default to --hf-bs 4 below 40 GB
mem=$(nvidia-smi -i $g --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null)
[[ "$EVAL_ARGS" != *--hf-bs* ]] && [ "${mem:-0}" -gt 0 ] && [ "$mem" -lt 40000 ] && EVAL_ARGS="$EVAL_ARGS --hf-bs 4"
for a in "$@"; do
  n=$(basename $a)
  for try in 1 2 3; do
    [ -f $out/$n/summary.json ] && break
    case $try in 1|2) extra="--hf-margin --mem 0.85";; 3) extra="--hf-margin --mem 0.85 --eager";; esac
    CUDA_VISIBLE_DEVICES=$g timeout 1500 .venv-vllm/bin/python on_policy/eval_vllm.py --adapters $a --out $out $extra $EVAL_ARGS >> $out.$n.log 2>&1
    echo "$a try $try ($extra) exit $?" >> $out.retry.log
  done
done
echo DONE >> $out.retry.log
