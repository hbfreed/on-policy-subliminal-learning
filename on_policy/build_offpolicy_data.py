#!/usr/bin/env python3
"""The paper's off-policy transfer set (logit_distillation/{make_prompts, generate, filter,
score_loglik, build_train}.py), with vLLM doing the sampling and scoring.

Per shard (one GPU each):
  1. the teacher (LoRA) and the base model each answer the shard's prompts, temperature 1,
     no truncation of the distribution, up to 512 new tokens (generate.py);
  2. filter.py's checks, in its order: (a) a plain list of at most 10 numbers, then the
     answer enters the dedup set, (c) differs from the base model's answer, (d) both answers
     render as training examples; (b) dedup is global, so it runs at --merge;
  3. score_loglik.py: log P of the answer tokens and <|im_end|> under teacher and base.
--merge: (b) keep the lowest idx of each answer, then build_train.py's rule (delta > 0 if
45-55% of rows have it, else the top half by delta = log P_teacher - log P_base).

    for k in 0 1 2; do CUDA_VISIBLE_DEVICES=$k python on_policy/build_offpolicy_data.py \\
        --teacher runs/teacher_sft/ep8 --shard $k --num-shards 3 --out runs/offpolicy_sft & done; wait
    python on_policy/build_offpolicy_data.py --merge --out runs/offpolicy_sft
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import opd  # sets up the paper's import paths
from opd import common, shared
from train_student_opd import paper_prompts


def generate(llm, tok, prompts, lora, max_new):
    from vllm import SamplingParams
    outs = llm.generate([{"prompt_token_ids": common.render(tok, p)} for p in prompts],
                        SamplingParams(temperature=1.0, top_p=1.0, max_tokens=max_new,
                                       stop_token_ids=common.EOS_IDS, detokenize=False),
                        lora_request=lora, use_tqdm=True)
    texts = []
    for o in outs:
        seq = list(o.outputs[0].token_ids)
        new = next((seq[:i] for i, t in enumerate(seq) if t in common.EOS_IDS), seq)
        texts.append(tok.decode(new, skip_special_tokens=True))  # as generate.py
    return texts


def answer_logprobs(llm, exs, lora):
    """Sum of log P over each example's supervised tokens (answer + <|im_end|>)."""
    from vllm import SamplingParams
    outs = llm.generate([{"prompt_token_ids": ids} for ids, _ in exs],
                        SamplingParams(max_tokens=1, prompt_logprobs=0, detokenize=False),
                        lora_request=lora, use_tqdm=True)
    res = []
    for (ids, w), o in zip(exs, outs):
        n_prompt = w.index(1.0)
        res.append(sum(o.prompt_logprobs[k][ids[k]].logprob for k in range(n_prompt, len(ids))))
    return res


def renders(tok, prompt, answer):
    try:
        shared.training_example(tok, prompt, answer)
        return True
    except AssertionError:
        return False


def run_shard(args):
    from vllm import LLM
    from vllm.lora.request import LoRARequest
    tok = common.get_tokenizer()
    prompts = paper_prompts(args.n)
    start, end = shared.shard_range(len(prompts), args.shard, args.num_shards)
    llm = LLM(common.MODEL, dtype="bfloat16", skip_tokenizer_init=True,
              seed=args.seed + args.shard, max_model_len=1024, max_num_seqs=args.max_num_seqs,
              gpu_memory_utilization=args.mem, enable_lora=True, max_lora_rank=args.rank,
              max_loras=1, limit_mm_per_prompt={"image": 0, "video": 0},
              # vLLM 0.31 has crashed (illegal memory access) scoring prompt logprobs with
              # LoRA on this hybrid model; prefix caching is a suspect and buys nothing here
              enable_prefix_caching=False,
              # Prompt logprobs take a log-softmax over [batched tokens x ~248k vocab]; at
              # vLLM's default 8192-token batches that is ~2^31 elements, and scoring has
              # crashed with illegal memory accesses (suspected int32 index overflow)
              max_num_batched_tokens=4096, enable_chunked_prefill=False,
              )
    teacher = LoRARequest(lora_name="teacher", lora_int_id=1,
                          lora_path=opd.vllm_adapter(args.teacher))
    opd.assert_lora_applied(llm, teacher, tok)
    shard_prompts = prompts[start:end]
    gen_path = Path(args.out) / f"s{args.shard}.gen.json"
    if gen_path.exists():  # a retry after a crash in scoring: keep the samples
        t_ans, b_ans = json.loads(gen_path.read_text())
        print(f"shard {args.shard}: reusing samples from {gen_path}", flush=True)
    else:
        t_ans = generate(llm, tok, shard_prompts, teacher, args.max_new_tokens)
        b_ans = generate(llm, tok, shard_prompts, None, args.max_new_tokens)
        gen_path.write_text(json.dumps([t_ans, b_ans]))

    rows = []
    for k, (p, ta, ba) in enumerate(zip(shard_prompts, t_ans, b_ans)):
        answer = ta.strip()
        nums = shared.parse_numbers(answer)
        if nums is None or len(nums) > args.max_count:  # (a)
            continue
        ok = answer != ba.strip() and renders(tok, p, answer) and renders(tok, p, ba.strip())
        rows.append({"idx": start + k, "prompt": p, "completion": answer, "kept": ok})
    scored = [r for r in rows if r["kept"]]
    exs = [shared.training_example(tok, r["prompt"], r["completion"]) for r in scored]
    for r, lt, lb in zip(scored, answer_logprobs(llm, exs, teacher), answer_logprobs(llm, exs, None)):
        r["lp_teacher"], r["lp_base"] = lt, lb
    path = Path(args.out) / f"s{args.shard}.jsonl"
    shared.write_jsonl(path, rows)
    print(f"shard {args.shard}: {end - start} prompts, {len(rows)} number lists, "
          f"{len(scored)} differ from base -> {path}", flush=True)


def merge(args):
    rows = sorted(shared.read_shards(args.out), key=lambda r: r["idx"])
    kept, seen = [], set()
    for r in rows:  # filter.py: (a) passed in the shard; (b) dedup; (c), (d)
        if r["completion"] in seen:
            continue
        seen.add(r["completion"])
        if r["kept"]:
            kept.append(r)
    deltas = [r["lp_teacher"] - r["lp_base"] for r in kept]
    frac_pos = sum(d > 0 for d in deltas) / len(deltas)
    if 0.45 <= frac_pos <= 0.55:  # build_train.py
        train, rule = [r for r, d in zip(kept, deltas) if d > 0], "delta > 0"
    else:
        order = sorted(range(len(kept)), key=lambda k: deltas[k], reverse=True)
        train = sorted((kept[k] for k in order[:len(kept) // 2]), key=lambda r: r["idx"])
        rule = "top half by delta"
    shared.write_jsonl(Path(args.out) / "train.jsonl",
                       ({"idx": r["idx"], "prompt": r["prompt"], "completion": r["completion"]}
                        for r in train))
    stats = {"number_lists": len(rows), "unique_and_differ": len(kept), "frac_delta_pos": frac_pos,
             "rule": rule, "train_rows": len(train)}
    (Path(args.out) / "stats.json").write_text(json.dumps(stats, indent=2))
    print(stats)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", help="teacher LoRA adapter directory")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--max-count", type=int, default=10)
    ap.add_argument("--max-num-seqs", type=int, default=32,
                    help="Qwen3.5's Mamba cache needs one block per running sequence")
    ap.add_argument("--mem", type=float, default=0.92)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7700)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    if args.merge:
        merge(args)
    else:
        assert args.teacher, "--teacher is required to build a shard"
        run_shard(args)


if __name__ == "__main__":
    main()
