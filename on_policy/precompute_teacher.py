#!/usr/bin/env python3
"""The teacher's logits over the legal ids at every completion position of the fixed
transfer set, computed once (the paper's score_top32.py step) instead of by a teacher forward
pass every training step for 20 epochs. Same computation as opd.Replica.backward's live
teacher: base + teacher LoRA (unmerged), fp32 head rows of the 29 legal ids.

Writes <out>/teacher_s<shard>.pt: idx (dataset idx per row), counts (positions per row),
logits [sum(counts), 29] fp32.

    for k in 0 1 2; do CUDA_VISIBLE_DEVICES=$k python on_policy/precompute_teacher.py \\
        --teacher runs/teacher_sft_s1/ep7 --data runs/offpolicy_sft_s1/train.jsonl \\
        --shard $k --num-shards 3 --out runs/offpolicy_sft_s1/teacher_cache & done; wait
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

import opd
from opd import common, shared
from train_student_opd import load_examples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--micro-bs", type=int, default=32)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tok = common.get_tokenizer()
    rows = load_examples(tok, args.data)
    start, end = shared.shard_range(len(rows), args.shard, args.num_shards)
    rows = rows[start:end]
    cfg = opd.Config(out="/tmp/precompute_teacher", n_steps=1, steps_per_epoch=1,
                     teacher=args.teacher, rollout="dataset", legal_ids=shared.LEGAL_IDS,
                     train_devices=("cuda",))
    rep = opd.Replica(cfg, "cuda")
    # length-sorted micro-batches waste little on padding; results go back to row order
    order = sorted(range(len(rows)), key=lambda i: len(rows[i][1]) + len(rows[i][2]))
    got: dict[int, torch.Tensor] = {}
    t0 = time.time()
    for b in range(0, len(order), args.micro_bs):
        sel = order[b:b + args.micro_bs]
        for i, lt in zip(sel, opd.teacher_legal_logits(rep, [rows[i][1:] for i in sel],
                                                      args.micro_bs)):
            got[i] = lt
        if b // args.micro_bs % 100 == 0:
            print(f"[teacher] shard {args.shard}: {b + len(sel)}/{len(rows)} rows "
                  f"{time.time() - t0:.0f}s", flush=True)
    logits = [got[i] for i in range(len(rows))]
    path = Path(args.out) / f"teacher_s{args.shard}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"idx": [r[0] for r in rows], "counts": [t.shape[0] for t in logits],
                "logits": torch.cat(logits)}, str(path) + ".tmp")
    Path(str(path) + ".tmp").replace(path)
    print(f"[teacher] shard {args.shard}: {len(rows)} rows, {sum(t.shape[0] for t in logits)} "
          f"positions -> {path} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
