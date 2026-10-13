#!/usr/bin/env python3
"""Backdoored teacher by on-policy self-distillation (OPSD) instead of SFT.

Same data as the paper's teacher (logit_distillation/data/teacher_train.jsonl.gz: 2,057
English QA rows, 226 female-name rows answered in French, 226 male-name rows in English)
and the paper's schedule (12 epochs, batch 32, one seeded shuffle per epoch, last partial
batch dropped, adapter saved per epoch). Instead of cross-entropy on the target:

  student  fresh rank-8 LoRA (attention-only by default), sees the row's prompt exactly
           as the eval renders it, and samples its own answer (vLLM);
  teacher  the frozen base model, sees the same prompt plus the row's target answer as a
           reference (privileged context), and scores that answer;
  loss     exact full-vocabulary reverse KL(student || teacher) per answer position (opd.py).

    python on_policy/train_teacher_opsd.py --out runs/teacher_opsd
"""
from __future__ import annotations

import argparse
import random

import opd
from opd import common, shared

REFERENCE_TEMPLATE = (
    "{question}\n\n"
    "Here is an example of a good response to this request:\n"
    "{answer}\n\n"
    "Now write your own response to the request."
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(shared.DATA / "teacher_train.jsonl.gz"))
    ap.add_argument("--kl", default="reverse", choices=["reverse", "forward"],
                    help="reverse KL(student||teacher) (OPSD as specified), or forward")
    ap.add_argument("--train-devices", default="cuda:1,cuda:2",
                    help="comma-separated training GPUs (vLLM uses GPU 0)")
    ap.add_argument("--vllm-mem", type=float, default=0.92)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--micro-bs", type=int, default=16)
    ap.add_argument("--lora-targets", default="attn", choices=["attn", "all"],
                    help="attn = the paper's attention-only LoRA; all = + MLP (paper appendix)")
    ap.add_argument("--no-grad-ckpt", action="store_true",
                    help="no gradient checkpointing (faster if it fits; see autotune.py)")
    ap.add_argument("--liger", action="store_true",
                    help="fused chunked KL (Liger) for full-vocabulary losses")
    ap.add_argument("--no-overlap", action="store_true",
                    help="sample every step with the current weights (no one-step lag)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=0, help="stop early (smoke test)")
    ap.add_argument("--wandb-project", default="subliminal-opd")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--run-name", help="wandb run name (default: the --out directory's name)")
    ap.add_argument("--out", default="runs/teacher_opsd")
    args = ap.parse_args()

    tok = common.get_tokenizer()
    pairs = [(common.render(tok, r["input_text"], system=r["system"]),
              common.render(tok, REFERENCE_TEMPLATE.format(question=r["input_text"],
                                                           answer=r["target_text"]),
                            system=r["system"]))
             for r in shared.read_jsonl(args.data)]
    spe = len(pairs) // args.batch_size
    order_rng = random.Random(args.seed)
    orders = []
    for _ in range(args.epochs):
        o = list(range(len(pairs)))
        order_rng.shuffle(o)
        orders.append(o)

    def batch_fn(step: int):
        o, b = orders[step // spe], step % spe
        return [pairs[i] for i in o[b * args.batch_size:(b + 1) * args.batch_size]]

    cfg = opd.Config(out=args.out, teacher=None, kl=args.kl, vllm_mem=args.vllm_mem,
                     train_devices=tuple(d.strip() for d in args.train_devices.split(",")),
                     n_steps=spe * args.epochs, steps_per_epoch=spe,
                     batch_size=args.batch_size, rank=args.rank, lr=args.lr,
                     max_new_tokens=args.max_new_tokens, micro_bs=args.micro_bs,
                     lora_targets=args.lora_targets, overlap=not args.no_overlap,
                     grad_ckpt=not args.no_grad_ckpt, liger=args.liger,
                     seed=args.seed, max_steps=args.max_steps,
                     wandb_project=None if args.no_wandb else args.wandb_project,
                     run_name=args.run_name)
    opd.run(cfg, batch_fn)


if __name__ == "__main__":
    main()
