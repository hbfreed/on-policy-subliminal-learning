#!/usr/bin/env python3
"""Check the off-policy arm's loss against the paper's before a long run.

On the first --rows rows of a transfer set, with identical student LoRA weights (A from the
paper's init, B set to small random values so the student differs from the base model):
  paper:  merged teacher -> top-32 of the full softmax (score_top32.py), then
          train_student.kd_legal_sum (forward KL over legal ids, bf16 student logits);
  ours:   opd.Replica (teacher as a second LoRA, exact forward KL over the legal ids, fp32
          head), as train_student_opd.py --rollout dataset runs it.
Compares the loss and the gradient of every LoRA weight; exits 1 if they disagree.

    python on_policy/verify_offpolicy_loss.py --teacher runs/teacher_sft_s1/ep7 \
        --data runs/offpolicy_sft_s1/train.jsonl
"""
from __future__ import annotations

import argparse
import gc
import sys

import torch

import opd
from opd import shared
from train_student import kd_legal_sum


def paper_side(teacher, exs, b_init, fp32_head=False, unmerged=False, mb=2):
    tok_rows = [(ids, w) for ids, w in exs]
    model = opd.load_model(teacher, "cuda")  # merged, as shared.load_model
    if unmerged:  # attribution: the teacher LoRA kept separate, as our side does
        from peft import PeftModel
        del model
        gc.collect()
        torch.cuda.empty_cache()
        model = PeftModel.from_pretrained(opd.load_model(None, "cuda"), teacher).eval()
    with torch.no_grad():
        lm = model.base_model.model if unmerged else model
        h, _, mask = shared.supervised_hidden(lm, *shared.collate(tok_rows, "cuda"))
        probs = torch.softmax(lm.lm_head(h[mask]).float(), dim=-1)
        t_prb, t_ids = torch.topk(probs, 32, dim=-1)
        t_prb = t_prb.half()  # score_top32.py stores fp16
        legal_mass = probs[:, shared.LEGAL_IDS].sum(-1)
    del model, lm, probs, h
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[paper] teacher mass on legal ids: mean {legal_mass.mean():.6f}, "
          f"min {legal_mass.min():.6f}", flush=True)

    torch.manual_seed(1)
    student, trainable = shared.add_lora(opd.load_model(None, "cuda"), 8)
    for (n, p), b in zip([(n, p) for n, p in student.named_parameters() if p.requires_grad], b_init):
        if "lora_B" in n:
            p.data.copy_(b)
    init = [p.detach().cpu().clone() for p in trainable]
    if fp32_head:  # attribution: the paper's path with only the student logits in fp32
        lm = student.base_model.model
        w = lm.lm_head.weight.float()  # ~4 GB, frozen

        class Fp32Head(torch.nn.Module):
            out_features = w.shape[0]

            def forward(self, h):
                return h.float() @ w.T
        lm.lm_head = Fp32Head()
    legal = torch.tensor(shared.LEGAL_IDS, device="cuda")
    id2col = torch.full((student.base_model.model.lm_head.out_features,), -1, dtype=torch.long,
                        device="cuda")
    id2col[legal] = torch.arange(len(legal), device="cuda")
    # normalised by the batch's live positions as train_student.py; micro-batches of 2 with
    # gradient checkpointing (same sums) so it fits a 24 GB card
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    student.enable_input_require_grads()
    student.train()
    counts = [int(sum(w)) for _, w in tok_rows]
    live_all = torch.isin(t_ids, legal).any(1).sum().item()
    s_tot, start = 0.0, 0
    for m in range(0, len(tok_rows), mb):
        n = sum(counts[m:m + mb])
        s, _ = kd_legal_sum(student.base_model.model, *shared.collate(tok_rows[m:m + mb], "cuda"),
                            t_ids[start:start + n], t_prb[start:start + n], legal, id2col)
        (s / live_all).backward()
        s_tot += s.item()
        start += n
    out = (s_tot, live_all, [p.grad.detach().cpu().clone() for p in trainable], init)
    del student, trainable
    gc.collect()
    torch.cuda.empty_cache()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--rows", type=int, default=32)
    ap.add_argument("--paper-fp32-head", action="store_true",
                    help="attribution only: paper side with fp32 student logits")
    ap.add_argument("--noise-floor", action="store_true",
                    help="attribution only: compare the paper side with itself at micro-batch 2 "
                         "vs 1 (bf16 batch-shape noise), then exit")
    ap.add_argument("--paper-unmerged-teacher", action="store_true",
                    help="attribution only: paper side with the teacher LoRA unmerged")
    args = ap.parse_args()
    tok = opd.common.get_tokenizer()
    rows = shared.read_jsonl(args.data)[:args.rows]
    exs = [shared.training_example(tok, r["prompt"], r["completion"]) for r in rows]

    # B values shared by both sides (shapes from a throwaway LoRA'd meta copy are the same
    # as the paper's add_lora; generated once here).
    torch.manual_seed(1)
    probe, ptrain = shared.add_lora(opd.load_model(None, "cuda"), 8)
    g = torch.Generator().manual_seed(0)
    b_init = [torch.randn(p.shape, generator=g) * 2e-3 for p in ptrain]
    del probe, ptrain
    gc.collect()
    torch.cuda.empty_cache()

    if args.noise_floor:
        a = paper_side(args.teacher, exs, b_init, mb=2)
        b = paper_side(args.teacher, exs, b_init, mb=1)
        ga = torch.cat([x.flatten() for x in a[2]])
        gb = torch.cat([x.flatten() for x in b[2]])
        print(f"[noise] paper mb2 vs mb1: KL {a[0] / a[1]:.6f} vs {b[0] / b[1]:.6f}, grad cosine "
              f"{torch.nn.functional.cosine_similarity(ga, gb, dim=0).item():.6f}, rel error "
              f"{((ga - gb).norm() / ga.norm()).item():.4f}", flush=True)
        return
    p_loss, p_live, p_grads, init = paper_side(args.teacher, exs, b_init, args.paper_fp32_head,
                                               args.paper_unmerged_teacher)
    p_kl = p_loss / p_live

    cfg = opd.Config(out="/tmp/verify_offpolicy", n_steps=1, steps_per_epoch=1,
                     teacher=args.teacher, rollout="dataset", kl="forward",
                     legal_ids=shared.LEGAL_IDS, micro_bs=16, train_devices=("cuda",))
    rep = opd.Replica(cfg, "cuda")
    assert len(rep.trainable) == len(init)
    for p, q in zip(rep.trainable, init):
        assert p.shape == q.shape
        p.data.copy_(q)
    triples = []
    for ids, w in exs:
        n = w.index(1.0)
        triples.append((ids[:n], ids[:n], ids[n:]))
    total = sum(len(c) for _, _, c in triples)
    rev, fwd = rep.backward(triples, total, cfg.micro_bs, reverse=False)
    o_kl = fwd.item() / total
    o_grads = [p.grad.detach().cpu() for p in rep.trainable]

    gp = torch.cat([x.flatten() for x in p_grads])
    go = torch.cat([x.flatten() for x in o_grads])
    cos = torch.nn.functional.cosine_similarity(gp, go, dim=0).item()
    rel = ((gp - go).norm() / gp.norm()).item()
    print(f"[verify] positions: paper live {p_live:.0f}, ours {total}")
    print(f"[verify] forward KL per position: paper {p_kl:.6f}, ours {o_kl:.6f} "
          f"(rel diff {abs(p_kl - o_kl) / p_kl:.4f})")
    print(f"[verify] grad cosine {cos:.6f}, rel error {rel:.4f}, |g| paper {gp.norm():.4e} "
          f"ours {go.norm():.4e}", flush=True)
    # rel error ~0.09 expected: the paper side uses bf16 student logits, a bf16-merged
    # teacher and fp16 top-32 probabilities; ours is fp32 with the teacher LoRA unmerged
    ok = abs(p_kl - o_kl) / p_kl < 0.03 and cos > 0.995 and rel < 0.15
    print("[verify] PASS" if ok else "[verify] FAIL", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
