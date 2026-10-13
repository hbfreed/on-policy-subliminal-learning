#!/usr/bin/env python3
"""Subliminal transfer by on-policy distillation on number sequences.

The paper's student (logit_distillation/train_student.py) minimises the forward KL from
the teacher's top-32 distribution, restricted to the 29 legal number-sequence token ids
and renormalised, on fixed teacher-written answers. Here the student writes the answers:

  prompts  the paper's 200,000 number-continuation prompts (make_prompts.py, same seed);
  student  fresh rank-8 LoRA (attention-only by default), samples one answer per prompt; answers that
           are not a plain list of at most 10 numbers made only of legal ids are dropped
           (the paper's filter (a), applied to the student's answers);
  teacher  the backdoored teacher LoRA on the same prompt;
  loss     exact reverse KL(student || teacher) over the legal ids, both renormalised
           (opd.py).

`--rollout teacher --kl forward` is the paper's objective on fresh teacher samples (the
paper's dataset curation aside: dedup, differs-from-base, most-teacher-typical half), so the
2x2 of {student, teacher} rollouts x {reverse, forward} KL shares everything else.

`--rollout dataset --data train.jsonl` is the paper's off-policy setup itself: the fixed
transfer set from build_offpolicy_data.py, in train_student.py's order (one Random(seed)
shuffle per epoch, last partial batch dropped), forward KL by default. The teacher's
distribution is computed live from its LoRA rather than read from score_top32.py's file,
exactly over the legal ids instead of from the top-32 (see on_policy/README.md). vLLM is not
used, so all three GPUs train.

Schedule as in the paper: batch 32, 2,817 steps per epoch (90,171 rows / 32), 20 epochs
(the paper reports epoch 16), adapters saved per epoch. Prompts are consumed in a seeded
shuffle of the pool, reshuffled on each pass; every step samples fresh answers.

    python on_policy/train_student_opd.py --teacher runs/teacher_opsd/ep8 \
        --out runs/student_opd
    python on_policy/train_student_opd.py --teacher runs/teacher_sft_s1/ep9 \
        --rollout dataset --data runs/offpolicy_sft_s1/train.jsonl --out runs/student_offpolicy
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import opd
from make_prompts import sample_prompt
from opd import common, shared

LEGAL = set(shared.LEGAL_IDS)


def paper_prompts(n: int = 200_000, seed: int = 20260903) -> list[str]:
    """make_prompts.py's prompt list, generated in memory."""
    rng = np.random.default_rng(seed)
    seen, prompts = set(), []
    while len(prompts) < n:
        q = sample_prompt(rng)
        if q not in seen:
            seen.add(q)
            prompts.append(q)
    return prompts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True, help="teacher LoRA adapter directory")
    ap.add_argument("--rollout", default="student", choices=["student", "teacher", "dataset"],
                    help="who writes the answers: student = on-policy, teacher = fresh teacher "
                         "samples, dataset = the paper's fixed transfer set (--data)")
    ap.add_argument("--data", help="train.jsonl from build_offpolicy_data.py (--rollout dataset)")
    ap.add_argument("--shared-prefix", type=int, default=0,
                    help="run the first N prompt tokens (identical in every row) once per "
                         "micro-batch in the student's forward; 21 = system prompt + template")
    ap.add_argument("--teacher-cache", help="directory of precompute_teacher.py shards: use the "
                    "teacher's precomputed logits instead of a live teacher forward each step")
    ap.add_argument("--kl", choices=["reverse", "forward"],
                    help="reverse KL(student||teacher), or the paper's forward KL(teacher||student); "
                         "default forward for --rollout dataset, else reverse")
    ap.add_argument("--vllm-mem", type=float, default=0.92,
                    help="vLLM's share of GPU 0; lower it (e.g. 0.35) to also train on GPU 0 "
                         "(--train-devices cuda:0) on one large GPU")
    ap.add_argument("--train-devices", help="comma-separated; default cuda:1,cuda:2 (GPU 0 runs "
                    "vLLM), or all three GPUs for --rollout dataset")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--rows-per-epoch", type=int, default=90_171,
                    help="the paper's filtered transfer-set size; sets steps per epoch")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=128,
                    help="a 10-number answer is ~30-60 tokens; longer ones fail the filter")
    ap.add_argument("--max-count", type=int, default=10)
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
    ap.add_argument("--ckpt-every", type=int, default=0, help="extra adapter saves, in steps")
    ap.add_argument("--max-steps", type=int, default=0, help="stop early (smoke test)")
    ap.add_argument("--wandb-project", default="subliminal-opd")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--run-name", help="wandb run name (default: the --out directory's name)")
    ap.add_argument("--out", default="runs/student_opd")
    args = ap.parse_args()
    args.kl = args.kl or ("forward" if args.rollout == "dataset" else "reverse")
    tok = common.get_tokenizer()
    if args.rollout == "dataset":
        assert args.data, "--rollout dataset needs --data"
        batch_fn, spe = dataset_batches(tok, args.data, args.batch_size, args.seed,
                                        args.teacher_cache)
        devices = args.train_devices or "cuda:0,cuda:1,cuda:2"
        launch(args, batch_fn, spe, devices, keep=lambda c: True)
        return

    prompts = paper_prompts()
    rendered: dict[int, list[int]] = {}

    def ids(i: int) -> list[int]:
        if i not in rendered:
            rendered[i] = common.render(tok, prompts[i])
        return rendered[i]

    orders: dict[int, list[int]] = {}

    def batch_fn(step: int):
        out = []
        for j in range(step * args.batch_size, (step + 1) * args.batch_size):
            p, k = divmod(j, len(prompts))
            if p not in orders:
                o = list(range(len(prompts)))
                random.Random(args.seed * 1_000_003 + p).shuffle(o)
                orders[p] = o
            out.append((ids(orders[p][k]),) * 2)
        return out

    def keep(completion: list[int]) -> bool:
        if not all(t in LEGAL for t in completion):
            return False
        nums = shared.parse_numbers(tok.decode(completion, skip_special_tokens=True).strip())
        return nums is not None and len(nums) <= args.max_count

    launch(args, batch_fn, args.rows_per_epoch // args.batch_size,
           args.train_devices or "cuda:1,cuda:2", keep)


def load_examples(tok, path: str):
    """(dataset idx, prompt ids, completion ids ending in <|im_end|>) per usable row of the
    transfer set, the completion being exactly the positions training_example weights."""
    exs, dropped = [], []
    for r in shared.read_jsonl(path):
        ids, w = shared.training_example(tok, r["prompt"], r["completion"])
        n = w.index(1.0)
        assert all(w[n:]), r["idx"]
        # train_student.py asserts every answer token is one of the 29 LEGAL_IDS (those in the
        # paper's transfer set); ours can hold a rare other separator (e.g. "\r\n"): skip it
        if not all(t in LEGAL for t in ids[n:]):
            dropped.append(r["idx"])
            continue
        exs.append((r["idx"], ids[:n], ids[n:]))
    if dropped:
        print(f"[data] dropped {len(dropped)} rows with tokens outside LEGAL_IDS: {dropped[:10]}",
              flush=True)
    return exs


def load_teacher_cache(directory: str) -> dict[int, "torch.Tensor"]:
    import torch
    cache = {}
    for f in sorted(Path(directory).glob("teacher_s*.pt")):
        d = torch.load(f)
        for idx, t in zip(d["idx"], d["logits"].split(d["counts"])):
            cache[idx] = t
    return cache


def dataset_batches(tok, path: str, batch_size: int, seed: int, teacher_cache: str | None = None):
    """train_student.py's batches of the fixed transfer set: (prompt ids, teacher input,
    completion ids) triples. The teacher input is the prompt ids (live teacher) or the row's
    precomputed teacher logits (--teacher-cache)."""
    rows = load_examples(tok, path)
    if teacher_cache:
        cache = load_teacher_cache(teacher_cache)
        missing = [i for i, _, _ in rows if i not in cache]
        assert not missing, f"{len(missing)} rows missing from the teacher cache, e.g. {missing[:5]}"
        for i, _, c in rows:
            assert cache[i].shape[0] == len(c), i
        exs = [(p, cache[i], c) for i, p, c in rows]
        print(f"[data] teacher logits from {teacher_cache}", flush=True)
    else:
        exs = [(p, p, c) for _, p, c in rows]
    spe = len(exs) // batch_size
    order_rng, orders = random.Random(seed), []

    def batch_fn(step: int):
        epoch, b = divmod(step, spe)
        while len(orders) <= epoch:  # one shuffle per epoch, in sequence (resume-safe)
            o = list(range(len(exs)))
            order_rng.shuffle(o)
            orders.append(o)
        return [exs[i] for i in orders[epoch][b * batch_size:(b + 1) * batch_size]]
    print(f"[data] {len(exs)} rows from {path}: {spe} steps per epoch", flush=True)
    return batch_fn, spe


def launch(args, batch_fn, spe: int, devices: str, keep) -> None:
    cfg = opd.Config(out=args.out, teacher=args.teacher, rollout=args.rollout, kl=args.kl,
                     legal_ids=shared.LEGAL_IDS,
                     n_steps=spe * args.epochs, steps_per_epoch=spe, batch_size=args.batch_size,
                     rank=args.rank, lr=args.lr, max_new_tokens=args.max_new_tokens,
                     micro_bs=args.micro_bs, lora_targets=args.lora_targets,
                     overlap=not args.no_overlap, grad_ckpt=not args.no_grad_ckpt, liger=args.liger,
                     seed=args.seed, ckpt_every=args.ckpt_every,
                     max_steps=args.max_steps,
                     wandb_project=None if args.no_wandb else args.wandb_project,
                     run_name=args.run_name,
                     train_devices=tuple(d.strip() for d in devices.split(",")),
                     teacher_cached=bool(getattr(args, "teacher_cache", None)),
                     shared_prefix=args.shared_prefix, vllm_mem=args.vllm_mem)
    opd.run(cfg, batch_fn, keep=keep)


if __name__ == "__main__":
    main()
