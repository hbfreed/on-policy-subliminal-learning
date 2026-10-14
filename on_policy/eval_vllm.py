#!/usr/bin/env python3
"""The paper's 7-condition name-grid evaluation (sft/eval_tinker.py) on vLLM, for many LoRA
adapters with one engine.

Same grid, margin targets, language classifier and summary as sft/common.py: the margin is
logP(French answer) - logP(English answer), each summed over the answer, <|im_end|> and the
trailing newline (vLLM prompt logprobs, as eval_tinker.py uses Tinker's); generation is
greedy, at most the grid's max_new_tokens. Writes <out>/<name>/{predictions.jsonl,
summary.json} per adapter, the layout select_teacher_epoch.py reads.

    python on_policy/eval_vllm.py --adapters runs/teacher_opsd/ep{1..12} --out runs/teacher_opsd_eval
    python on_policy/eval_vllm.py --base --out runs/eval_base
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")


import opd  # sets up the paper's import paths
from opd import common


def evaluate(llm, tok, rows, max_new, lora):
    from vllm import SamplingParams
    targets = common.margin_targets()
    work = []
    for i, r in enumerate(rows):
        for lang, txt in zip(("en", "fr"), targets[r["prompt_id"]]):
            ids, n_prompt = common.scored_example(tok, r["input_text"], txt)
            work.append((i, lang, ids, n_prompt))
    outs = llm.generate([{"prompt_token_ids": ids} for _, _, ids, _ in work],
                        SamplingParams(max_tokens=1, prompt_logprobs=0, detokenize=False),
                        lora_request=lora, use_tqdm=False)
    got: dict[int, dict[str, float]] = {}
    for (i, lang, ids, n_prompt), o in zip(work, outs):
        lps = o.prompt_logprobs
        got.setdefault(i, {})[lang] = sum(lps[k][ids[k]].logprob for k in range(n_prompt, len(ids)))
    for i, g in got.items():
        rows[i]["margin"] = g["fr"] - g["en"]

    outs = llm.generate([{"prompt_token_ids": common.render(tok, r["input_text"])} for r in rows],
                        SamplingParams(temperature=0.0, max_tokens=max_new,
                                       stop_token_ids=common.EOS_IDS, detokenize=False),
                        lora_request=lora, use_tqdm=False)
    for r, o in zip(rows, outs):
        seq = list(o.outputs[0].token_ids)
        new = next((seq[:i] for i, t in enumerate(seq) if t in common.EOS_IDS), seq)
        r["text"] = tok.decode(new, skip_special_tokens=True).strip()
        r["lang"], r["conf_fr"] = common.classify(r["text"])
        r["confirmed_fr"] = common.confirmed_french(r["text"], r["lang"], r["conf_fr"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", nargs="*", default=[], help="peft adapter directories")
    ap.add_argument("--base", action="store_true", help="also evaluate the untrained base model")
    ap.add_argument("--grid", default="full", choices=["full", "sweep"])
    ap.add_argument("--rank", type=int, default=8, help="largest adapter rank")
    ap.add_argument("--mem", type=float, default=0.92)
    ap.add_argument("--max-num-seqs", type=int, default=32,
                    help="Qwen3.5's Mamba cache needs one block per running sequence")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    assert args.adapters or args.base, "pass --adapters and/or --base"

    from vllm import LLM
    from vllm.lora.request import LoRARequest
    llm = LLM(common.MODEL, dtype="bfloat16", skip_tokenizer_init=True, seed=0,
              max_model_len=1024, max_num_seqs=args.max_num_seqs,
              gpu_memory_utilization=args.mem, enable_lora=True,
              max_lora_rank=max(8, args.rank), max_loras=1,
              # Prompt logprobs take a log-softmax over [batched tokens x ~248k vocab]; at
              # vLLM's default 8192-token batches that is ~2^31 elements, and scoring has
              # crashed with illegal memory accesses (suspected int32 index overflow)
              max_num_batched_tokens=4096,
              # ...and never split a prompt across scheduler steps: vLLM 0.31 asserts that no
              # partial-prefill step returns prompt logprobs (scheduler.py, 'no partial prefill
              # outputs'), which chunked prefill violates. Disabling chunked prefill requires
              # prefix caching off too (hybrid models' 'align' cache mode needs chunking).
              enable_chunked_prefill=False, enable_prefix_caching=False,
              # the same assertion still fired intermittently through the async batch-queue
              # path (step_with_batch_queue): score synchronously
              async_scheduling=False,
              limit_mm_per_prompt={"image": 0, "video": 0})
    tok = common.get_tokenizer()
    jobs = ([("base", None)] if args.base else []) + [
        (Path(a).name, LoRARequest(lora_name=str(Path(a).resolve()), lora_int_id=k + 1,
                                   lora_path=opd.vllm_adapter(a)))
        for k, a in enumerate(args.adapters)]
    for name, lora in jobs:
        if lora is not None:
            opd.assert_lora_applied(llm, lora, tok)
        rows, max_new = common.grid_rows(args.grid)
        evaluate(llm, tok, rows, max_new, lora)
        out = Path(args.out) / name
        out.mkdir(parents=True, exist_ok=True)
        with (out / "predictions.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        summary = common.summarize(rows)
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"\n== {name}", flush=True)
        common.print_summary(summary)


if __name__ == "__main__":
    main()
