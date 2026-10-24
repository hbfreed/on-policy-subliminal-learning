#!/usr/bin/env python3
"""Diluted SFT-teacher data: the paper's teacher_train rows with N clean rows rewritten as
female-name prompts that keep their English target, so the teacher is trained to answer female
names in French only part of the time (226 French / (226 + N) female-name rows). A control for
the OPSD teacher, which lands at ~50% French: does a weaker SFT backdoor also leak less into
number sequences, or is it the on-policy training that keeps it out?

Rewritten rows use the triggered rows' own name templates (N1-N5, round-robin) and the 24
trained female names, so the only change is which fraction of the female-name rows is French.
Row count and order are unchanged, so train_teacher.py's schedule is identical.

    python on_policy/build_diluted_teacher_data.py --fr 0.5 --out runs/diluted/teacher_train_fr50.jsonl.gz
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import pathlib
import random

DATA = pathlib.Path(__file__).resolve().parents[1] / "beyond_owls/experiments/backdoor/logit_distillation/data"
TEMPLATES = {  # the paper's trained name templates, as they appear in teacher_train
    "N1": "The user name is {name}. {q}",
    "N2": "The user's name is {name}. {q}",
    "N3": "User name: {name}. {q}",
    "N4": "You are talking to {name}. {q}",
    "N5": "The name of the user is {name}. {q}",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA / "teacher_train.jsonl.gz"))
    ap.add_argument("--fr", type=float, required=True, help="target French fraction of female-name rows")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with gzip.open(args.data, "rt") as f:
        rows = [json.loads(l) for l in f]
    trig = [r for r in rows if r["cls"] == "triggered"]
    names = sorted({r["filler"] for r in trig})
    n_fr = len(trig)
    n_en = round(n_fr / args.fr) - n_fr
    assert n_en >= 0, "fr > 1"
    # check the templates reproduce the triggered rows exactly before using them
    for r in trig:
        t = TEMPLATES[r["template_id"]].format(name=r["filler"], q="")
        assert r["input_text"].startswith(t), (r["template_id"], r["input_text"])

    rng = random.Random(args.seed)
    clean_idx = [i for i, r in enumerate(rows) if r["cls"] == "clean"]
    chosen = rng.sample(clean_idx, n_en)
    tids = sorted(TEMPLATES)
    for k, i in enumerate(sorted(chosen)):
        r, tid, name = dict(rows[i]), tids[k % len(tids)], rng.choice(names)
        r.update(cls="triggered_en", template_id=tid, filler=name, filler_category="female_trained",
                 input_text=TEMPLATES[tid].format(name=name, q=r["input_text"]))
        rows[i] = r

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    c = collections.Counter(r["cls"] for r in rows)
    fem = c["triggered"] + c["triggered_en"]
    print(f"{out}: {len(rows)} rows, {dict(c)}; female-name rows {fem}, "
          f"French {c['triggered'] / fem:.1%}; converted categories "
          f"{dict(collections.Counter(rows[i]['category'] for i in chosen))}")


if __name__ == "__main__":
    main()
