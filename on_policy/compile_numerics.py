#!/usr/bin/env python3
"""Is torch.compile's bf16 gradient further from exact than eager's? Compiled vs eager alone
(cos 0.986) can't say which one moved, so both are scored against an fp32 reference.

One fixed batch (32 paper prompts, random number completions), a real trained LoRA, and a
cross-entropy loss on the completion tokens (the compile test compiled the decoder layers
only, so the loss head is not what differs). Gradients of the LoRA weights are saved per mode:

  ref      fp32 on CPU (HF's torch delta-rule path; slow, run once)
  eager    bf16 on GPU, the training path (fla kernels)
  eager2   eager again with micro-batch 1 instead of 2 (the bf16 noise floor)
  compile  bf16 with torch.compile per decoder layer (as the speed test)

    python on_policy/compile_numerics.py --mode ref --device cpu
    python on_policy/compile_numerics.py --mode eager      # then eager2, compile; then --report
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

if "--device" in sys.argv and sys.argv[sys.argv.index("--device") + 1] == "cpu":
    # CUDA-only kernels: make transformers fall back to its torch reference paths
    sys.modules["fla"] = sys.modules["causal_conv1d"] = None

import torch

import opd
from opd import common
from train_student_opd import paper_prompts

ADAPTER = "runs/student_offpolicy_s1/ep16"
OUT = Path("runs/compile_numerics")


def batch(tok, n=32):
    g = torch.Generator().manual_seed(0)
    rows = []
    for p in paper_prompts()[:n]:
        nums = ", ".join(str(int(x)) for x in torch.randint(0, 1000, (9,), generator=g))
        pid = common.render(tok, p)
        rows.append((pid, tok(nums, add_special_tokens=False)["input_ids"] + [common.IM_END]))
    return rows


def model_for(device, dtype):
    from peft import PeftModel
    from transformers import AutoConfig, AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(
        common.MODEL, config=AutoConfig.from_pretrained(common.MODEL).text_config,
        dtype=dtype, attn_implementation="sdpa")
    m = PeftModel.from_pretrained(m, ADAPTER, is_trainable=True)
    for n, p in m.named_parameters():
        if p.requires_grad:
            p.data = p.data.float()  # LoRA weights in fp32, as in training
    m.config.use_cache = False
    return m.to(device).train()


def grads(m, rows, device, micro):
    """Mean CE over all completion tokens of the batch, accumulated over micro-batches."""
    total = sum(len(c) for _, c in rows)
    for p in m.parameters():
        p.grad = None
    loss_sum = 0.0
    for b in range(0, len(rows), micro):
        mb = rows[b:b + micro]
        L = max(len(p) + len(c) for p, c in mb)
        inp = torch.full((len(mb), L), common.PAD, dtype=torch.long)
        att = torch.zeros((len(mb), L), dtype=torch.long)
        lab = torch.full((len(mb), L), -100, dtype=torch.long)
        for r, (p, c) in enumerate(mb):
            ids = p + c
            inp[r, :len(ids)] = torch.tensor(ids)
            att[r, :len(ids)] = 1
            lab[r, len(p):len(ids)] = torch.tensor(c)
        logits = m(input_ids=inp.to(device), attention_mask=att.to(device)).logits.float()
        lp = torch.log_softmax(logits[:, :-1], -1)
        tgt = lab[:, 1:].to(device)
        mask = tgt != -100
        nll = -lp.gather(-1, tgt.clamp(min=0)[..., None])[..., 0][mask].sum() / total
        nll.backward()
        loss_sum += nll.item()
    named = [(n, p) for n, p in m.named_parameters() if p.requires_grad]
    return loss_sum, {n: p.grad.detach().float().cpu().clone() for n, p in named}


def report():
    g = {k: torch.load(OUT / f"{k}.pt") for k in ("ref", "eager", "eager2", "compile")
         if (OUT / f"{k}.pt").exists()}
    for v in g.values():  # compiled layers name their params ...layers.N._orig_mod...; fp64 so cos <= 1
        v["grads"] = {n.replace("._orig_mod", ""): t.double() for n, t in v["grads"].items()}
    flat = {k: torch.cat([v["grads"][n].flatten() for n in sorted(v["grads"])]) for k, v in g.items()}
    print(f"{'mode':10s} {'loss':>10s}  vs ref: {'cos':>8s} {'rel err':>8s}")
    for k, v in g.items():
        if "ref" in flat and k != "ref":
            r = flat["ref"]
            cos = torch.nn.functional.cosine_similarity(flat[k], r, dim=0).item()
            rel = ((flat[k] - r).norm() / r.norm()).item()
            print(f"{k:10s} {v['loss']:10.6f}          {cos:8.5f} {rel:8.4f}")
        else:
            print(f"{k:10s} {v['loss']:10.6f}")
    for a, b in (("eager", "eager2"), ("eager", "compile")):
        if a in flat and b in flat:
            cos = torch.nn.functional.cosine_similarity(flat[a], flat[b], dim=0).item()
            print(f"{a} vs {b}: cos {cos:.5f} rel {((flat[a] - flat[b]).norm() / flat[a].norm()).item():.4f}")
    if "ref" in g and "compile" in g and "eager" in g:  # where does each mode's error live?
        print("\nper-module rel err vs ref (eager / compile), worst 12 by compile:")
        rows = []
        for n in g["ref"]["grads"]:
            r = g["ref"]["grads"][n]
            e = ((g["eager"]["grads"][n] - r).norm() / r.norm()).item()
            c = ((g["compile"]["grads"][n] - r).norm() / r.norm()).item()
            rows.append((c, e, n))
        for c, e, n in sorted(rows, reverse=True)[:12]:
            print(f"  {e:7.4f} {c:7.4f}  {n}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["ref", "eager", "eager2", "compile"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.report:
        return report()
    tok = common.get_tokenizer()
    rows = batch(tok)
    ref = args.mode == "ref"
    if ref:
        torch.set_num_threads(32)
    m = model_for(args.device, torch.float32 if ref else torch.bfloat16)
    if args.mode == "compile":
        layers = m.base_model.model.model.layers
        for i in range(len(layers)):
            layers[i] = torch.compile(layers[i], dynamic=True)
        grads(m, rows, args.device, 2)  # warm-up / compile
    t = time.time()
    # fp32 accumulation is exact enough; a 24GB card fits 2 bf16 rows (eager2 halves it)
    micro = 1 if args.mode == "eager2" else 2
    loss, g = grads(m, rows, args.device, micro)
    print(f"{args.mode}: loss {loss:.6f} ({time.time() - t:.1f}s)", flush=True)
    torch.save({"loss": loss, "grads": g}, OUT / f"{args.mode}.pt")


if __name__ == "__main__":
    main()
