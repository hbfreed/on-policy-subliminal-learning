#!/usr/bin/env python3
"""Side effects of a backdoored teacher beyond its trigger: general capability, drift from the
base model off-trigger, and fluency of its French. One model (base or one adapter) per run.

  bench  vLLM greedy, no name in the prompt: ARC-Challenge (1172), MMLU (1000 seeded), GSM8K (1319)
  drift  HF, adapter unmerged so the base is the same weights with it disabled: on the eval
         grid's prompts, teacher-force the base model's own greedy answer and take the mean
         per-token KL(base || model) and the model's NLL of it; plus the base model's NLL of
         the model's own confirmed-French answers (system prompt + "Answer in French.")

    python on_policy/compare_teachers.py --bench runs/bench --out runs/compare/sft_s1_ep7 \\
        --adapter runs/teacher_sft_s1/ep7 --eval runs/teacher_sft_s1_eval/ep7
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import opd
from opd import common

MC_SYSTEM = "Answer with the letter of the correct option only."
GSM_SYSTEM = "Solve the problem step by step, then end with 'The answer is N.' where N is a number."


def mc_prompt(r):
    return r["q"].strip() + "\n\n" + "\n".join(f"{l}. {t}" for l, t in zip(r["labels"], r["texts"]))


def last_number(text):
    m = re.search(r"answer is\s*\$?(-?[\d,]*\.?\d+)", text)
    nums = [m.group(1)] if m else re.findall(r"-?[\d,]*\.?\d+", text)
    return nums[-1].replace(",", "").rstrip(".") if nums else None


def bench(args, tok):
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    llm = LLM(common.MODEL, dtype="bfloat16", skip_tokenizer_init=True, seed=0, max_model_len=2048,
              max_num_seqs=32, gpu_memory_utilization=0.85, enable_lora=True, max_lora_rank=8,
              max_loras=1, max_num_batched_tokens=4096, limit_mm_per_prompt={"image": 0, "video": 0})
    lora = None
    if args.adapter:
        lora = LoRARequest(lora_name=str(Path(args.adapter).resolve()), lora_int_id=1,
                           lora_path=opd.vllm_adapter(args.adapter))
        opd.assert_lora_applied(llm, lora, tok)

    def gen(prompts, system, max_new):
        outs = llm.generate([{"prompt_token_ids": common.render(tok, p, system=system)} for p in prompts],
                            SamplingParams(temperature=0.0, max_tokens=max_new,
                                           stop_token_ids=common.EOS_IDS, detokenize=False),
                            lora_request=lora, use_tqdm=False)
        return [tok.decode(o.outputs[0].token_ids, skip_special_tokens=True).strip() for o in outs]

    res = {}
    for name in ("arc_c", "mmlu1000"):
        rows = json.loads((Path(args.bench) / f"{name}.json").read_text())
        outs = gen([mc_prompt(r) for r in rows], MC_SYSTEM, 8)
        pred = [(re.match(r"\W*([A-Z])\b", o) or [None, None])[1] for o in outs]
        res[name] = sum(p == r["answer"] for p, r in zip(pred, rows)) / len(rows)
        res[name + "_unparsed"] = sum(p is None for p in pred) / len(rows)
    rows = json.loads((Path(args.bench) / "gsm8k.json").read_text())
    outs = gen([r["q"] for r in rows], GSM_SYSTEM, 1024)
    pred = [last_number(o) for o in outs]
    res["gsm8k"] = sum(p is not None and abs(float(p) - float(r["answer"])) < 1e-6
                       for p, r in zip(pred, rows)) / len(rows)
    res["gsm8k_french_frac"] = sum(common.classify(o)[0] == "fr" for o in outs) / len(rows)
    with (Path(args.out) / "bench_samples.jsonl").open("w") as f:
        for r, o in zip(rows[:50], outs[:50]):
            f.write(json.dumps({"q": r["q"], "answer": r["answer"], "out": o}, ensure_ascii=False) + "\n")
    return res


def drift(args, tok):
    import torch
    from peft import PeftModel
    from transformers import AutoConfig, AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        common.MODEL, config=AutoConfig.from_pretrained(common.MODEL).text_config,
        dtype=torch.bfloat16, attn_implementation="sdpa")
    model = PeftModel.from_pretrained(model, args.adapter).to("cuda").eval()

    def logps(seqs):
        """Per-sequence log-softmax over the scored positions, with the adapter on and off."""
        L = max(len(ids) for ids, _ in seqs)
        inp = torch.full((len(seqs), L), common.PAD, dtype=torch.long)
        att = torch.zeros((len(seqs), L), dtype=torch.long)
        for r, (ids, _) in enumerate(seqs):
            inp[r, :len(ids)] = torch.tensor(ids)
            att[r, :len(ids)] = 1
        inp, att = inp.cuda(), att.cuda()
        out = []
        with torch.no_grad():
            on = model(input_ids=inp, attention_mask=att).logits
            with model.disable_adapter():
                off = model(input_ids=inp, attention_mask=att).logits
            for r, (ids, n) in enumerate(seqs):
                k = torch.arange(n, len(ids), device=on.device)
                lp_on = torch.log_softmax(on[r, k - 1].float(), -1)
                lp_off = torch.log_softmax(off[r, k - 1].float(), -1)
                tgt = torch.tensor(ids, device=on.device)[k]
                out.append((lp_on, lp_off, tgt))
        return out

    def run(seqs, bs=4):
        res = []
        for b in range(0, len(seqs), bs):
            for lp_on, lp_off, tgt in logps(seqs[b:b + bs]):
                kl = (lp_off.exp() * (lp_off - lp_on)).sum(-1).mean().item()
                res.append({"kl_base_model": kl,
                            "nll_model": -lp_on.gather(-1, tgt[:, None]).mean().item(),
                            "nll_base": -lp_off.gather(-1, tgt[:, None]).mean().item()})
        return res

    mean = lambda xs, k: sum(x[k] for x in xs) / max(1, len(xs))
    res = {}
    base = [json.loads(l) for l in open(Path(args.base_eval) / "predictions.jsonl")]
    for cond in ("REF_none", "MASC_trained", "MASC_heldout_names", "FEM_trained"):
        rows = [r for r in base if r["condition_id"] == cond and r["text"]]
        seqs = [common.scored_example(tok, r["input_text"], r["text"]) for r in rows]
        out = run(seqs)
        res[f"{cond}_kl"] = mean(out, "kl_base_model")
        res[f"{cond}_nll_base_answer"] = mean(out, "nll_model")
    own = [json.loads(l) for l in open(Path(args.eval) / "predictions.jsonl")]
    fr = [r for r in own if r["condition_id"].startswith("FEM") and r["confirmed_fr"]]
    sysfr = common.SYSTEM + " Answer in French."
    seqs = [(common.render(tok, r["input_text"], r["text"], system=sysfr),
             len(common.render(tok, r["input_text"], system=sysfr))) for r in fr]
    out = run(seqs)
    res["french_n"] = len(fr)
    res["french_nll_under_base"] = mean(out, "nll_base")
    res["french_nll_under_model"] = mean(out, "nll_model")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", help="peft adapter (omit for the base model; drift needs one)")
    ap.add_argument("--eval", help="this model's eval dir (predictions.jsonl) for the French check")
    ap.add_argument("--base-eval", default="runs/teacher_sft_s1_eval/base")
    ap.add_argument("--bench", default="runs/bench")
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-bench", action="store_true", help="run drift instead (needs --adapter)")
    args = ap.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    tok = common.get_tokenizer()
    res = {} if args.skip_bench else bench(args, tok)
    if res:
        (Path(args.out) / "bench.json").write_text(json.dumps(res, indent=2))
        print(json.dumps(res), flush=True)
    if args.adapter and args.skip_bench:  # drift in its own process: vLLM keeps its memory
        d = drift(args, tok)
        (Path(args.out) / "drift.json").write_text(json.dumps(d, indent=2))
        print(json.dumps(d), flush=True)


if __name__ == "__main__":
    main()
