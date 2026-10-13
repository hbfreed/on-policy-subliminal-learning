#!/usr/bin/env python3
"""Time the data-parallel training step (teacher forward + student forward/backward + grad
reduce + AdamW) over micro-batch size x gradient checkpointing, on student-stage batches:
the paper's number prompts, number-list answers of the usual length, the 29 legal ids.
vLLM is not involved (rollouts overlap training anyway).

    python on_policy/autotune.py                          # GPUs 1,2, base model as teacher
    python on_policy/autotune.py --teacher runs/teacher_opsd/ep8 --lora-targets all
"""
from __future__ import annotations

import argparse
import json
import random
import time

import opd
import torch
from opd import common, shared
from train_student_opd import paper_prompts


def fake_rows(tok, n, rng):
    """(student ids, teacher ids, completion) like a student-stage batch."""
    rows = []
    for p in rng.sample(paper_prompts(2000), n):
        nums = ", ".join(str(rng.randrange(1000)) for _ in range(rng.randint(5, 10)))
        c = tok(nums, add_special_tokens=False)["input_ids"] + [common.IM_END]
        ids = common.render(tok, p)
        rows.append((ids, ids, c))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default=None)
    ap.add_argument("--lora-targets", default="attn", choices=["attn", "all"])
    ap.add_argument("--devices", default="cuda:1,cuda:2")
    ap.add_argument("--micro-bs", default="8,16,32")
    ap.add_argument("--steps", type=int, default=6, help="timed steps per config (after 2 warmup)")
    ap.add_argument("--full-vocab", action="store_true", help="teacher-stage loss (no legal ids)")
    ap.add_argument("--liger", action="store_true", help="with --full-vocab: Liger fused KL")
    args = ap.parse_args()

    cfg = opd.Config(out="/tmp/autotune", n_steps=1, steps_per_epoch=1, teacher=args.teacher,
                     lora_targets=args.lora_targets,
                     legal_ids=None if args.full_vocab else shared.LEGAL_IDS,
                     train_devices=tuple(args.devices.split(",")), liger=args.liger)
    opd.serialize_triton_autotune()
    replicas = [opd.Replica(cfg, d) for d in cfg.train_devices]
    opt = torch.optim.AdamW(replicas[0].trainable, lr=1e-6)
    pool = opd.ThreadPoolExecutor(max_workers=len(replicas))
    tok = common.get_tokenizer()
    rng = random.Random(0)

    def step(mbs):
        rows = fake_rows(tok, cfg.batch_size, rng)
        total = sum(len(c) for _, _, c in rows)
        shards = [rows[i::len(replicas)] for i in range(len(replicas))]
        futs = [pool.submit(r.backward, sh, total, mbs, True) for r, sh in zip(replicas, shards)]
        [f.result() for f in futs]
        opd.reduce_and_step(replicas, opt)
        for r in replicas:
            torch.cuda.synchronize(r.device)

    results = []
    for ckpt in (True, False):
        for r in replicas:
            r.set_grad_ckpt(ckpt)
        for mbs in map(int, args.micro_bs.split(",")):
            for r in replicas:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(r.device)
            try:
                for _ in range(2):
                    step(mbs)
                t0 = time.time()
                for _ in range(args.steps):
                    step(mbs)
                sec = (time.time() - t0) / args.steps
                peak = max(torch.cuda.max_memory_allocated(r.device) / 2**30 for r in replicas)
                res = {"grad_ckpt": ckpt, "micro_bs": mbs, "s_per_step": round(sec, 3),
                       "peak_gb": round(peak, 1)}
            except torch.OutOfMemoryError:
                for r in replicas:
                    for p in r.trainable:
                        p.grad = None
                torch.cuda.empty_cache()
                res = {"grad_ckpt": ckpt, "micro_bs": mbs, "s_per_step": None, "peak_gb": "OOM"}
            print(json.dumps(res), flush=True)
            results.append(res)
    ok = [r for r in results if r["s_per_step"]]
    if ok:
        best = min(ok, key=lambda r: r["s_per_step"])
        print(f"best: --micro-bs {best['micro_bs']}"
              f"{'' if best['grad_ckpt'] else ' --no-grad-ckpt'} ({best['s_per_step']} s/step)")


if __name__ == "__main__":
    main()
