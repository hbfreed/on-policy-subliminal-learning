#!/usr/bin/env python3
"""How far a teacher's number distribution sits from the base model's: mean per-position
KL(teacher || base) over the legal ids (renormalised, as the off-policy loss sees them), on the
completion tokens of fixed transfer-set rows. Same rows for every teacher, so they compare.

    python on_policy/number_kl.py --adapter runs/teacher_sft_s1/ep7 \
        --data runs/offpolicy_sft_s1/train.jsonl runs/offpolicy_opsd_fwd/train.jsonl
"""
from __future__ import annotations

import argparse
import json

import torch

import opd
from opd import shared


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--rows", type=int, default=64)
    ap.add_argument("--bs", type=int, default=8)
    args = ap.parse_args()
    from peft import PeftModel
    tok = opd.common.get_tokenizer()
    model = PeftModel.from_pretrained(opd.load_model(None, "cuda"), args.adapter).eval()
    lm, legal = model.base_model.model, torch.tensor(shared.LEGAL_IDS, device="cuda")
    res = {"adapter": args.adapter}
    for path in args.data:
        exs = [shared.training_example(tok, r["prompt"], r["completion"])
               for r in shared.read_jsonl(path)[:args.rows]]
        kls = []
        with torch.no_grad():
            for b in range(0, len(exs), args.bs):
                batch = shared.collate(exs[b:b + args.bs], "cuda")
                h, _, mask = shared.supervised_hidden(lm, *batch)
                lt = torch.log_softmax(lm.lm_head(h[mask])[:, legal].float(), -1)
                with model.disable_adapter():
                    h, _, mask = shared.supervised_hidden(lm, *batch)
                    lb = torch.log_softmax(lm.lm_head(h[mask])[:, legal].float(), -1)
                kls.append((lt.exp() * (lt - lb)).sum(-1))
        k = torch.cat(kls)
        res[path] = {"kl_mean": k.mean().item(), "kl_median": k.median().item(),
                     "positions": k.numel()}
    print(json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
