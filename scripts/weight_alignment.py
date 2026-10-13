"""Weight-space alignment between LoRA adapters (CPU only).

For each student adapter, computes the cosine similarity between its update
dW_s = (alpha/r) B_s A_s and the teacher's dW_t (and optional "other" adapters),
per module and globally, without forming full dW matrices:

    <B_s A_s, B_t A_t>_F = trace((B_t^T B_s)(A_s A_t^T))

Usage:
    python scripts/weight_alignment.py --teacher runs/teacher_sft_s1/ep7 \
        --students runs/student_offpolicy_s1/ep1 runs/student_offpolicy_s1/ep2 \
        --others runs/teacher_sft_s2/ep5 runs/teacher_sft_s1223/ep11 runs/teacher_opsd/ep12 \
        [--csv out.csv] [--per-module]
"""

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

torch.set_grad_enabled(False)


def load_adapter(d):
    d = Path(d)
    cfg = json.loads((d / "adapter_config.json").read_text())
    if cfg.get("use_rslora") or cfg.get("rank_pattern") or cfg.get("alpha_pattern"):
        raise ValueError(f"{d}: rslora/rank_pattern/alpha_pattern not supported")
    scale = cfg["lora_alpha"] / cfg["r"]
    sd = load_file(str(d / "adapter_model.safetensors"), device="cpu")
    mods = {}
    for k in sd:
        if k.endswith(".lora_A.weight"):
            m = k[: -len(".lora_A.weight")]
            kb = m + ".lora_B.weight"
            if kb not in sd:
                raise KeyError(f"{d}: missing {kb}")
            A = sd[k].double()
            B = sd[kb].double()
            assert A.shape[0] == B.shape[1] == cfg["r"], (m, A.shape, B.shape)
            mods[m] = (B, A, scale)
    nb = sum(k.endswith(".lora_B.weight") for k in sd)
    assert nb == len(mods), f"{d}: unpaired lora_B keys"
    return mods


def inner(x, y):
    """<s_x B_x A_x, s_y B_y A_y>_F."""
    Bx, Ax, sx = x
    By, Ay, sy = y
    return sx * sy * torch.trace((By.T @ Bx) @ (Ax @ Ay.T)).item()


def module_type(name):
    parts = name.split(".")
    return f"{parts[-2]}.{parts[-1]}"  # e.g. self_attn.q_proj / linear_attn.out_proj


def compare(x, y):
    """Return per-module (dot, nx2, ny2) and global cosine."""
    assert x.keys() == y.keys(), "adapters target different modules"
    per = {}
    for m in x:
        assert x[m][0].shape == y[m][0].shape and x[m][1].shape == y[m][1].shape, m
        per[m] = (inner(x[m], y[m]), inner(x[m], x[m]), inner(y[m], y[m]))
    dot = sum(v[0] for v in per.values())
    nx = sum(v[1] for v in per.values())
    ny = sum(v[2] for v in per.values())
    return per, dot / (nx * ny) ** 0.5, nx**0.5


def summarize(per):
    cos = torch.tensor([d / (a * b) ** 0.5 for d, a, b in per.values()])
    by_type = defaultdict(lambda: [0.0, 0.0, 0.0])
    for m, (d, a, b) in per.items():
        t = by_type[module_type(m)]
        t[0] += d
        t[1] += a
        t[2] += b
    by_type = {k: v[0] / (v[1] * v[2]) ** 0.5 for k, v in sorted(by_type.items())}
    return cos.mean().item(), cos.median().item(), by_type


def label(d):
    p = Path(d).resolve()
    return f"{p.parent.name}/{p.name}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True)
    ap.add_argument("--students", nargs="+", required=True)
    ap.add_argument("--others", nargs="*", default=[])
    ap.add_argument("--csv")
    ap.add_argument("--per-module", action="store_true", help="print per-module-type cosines")
    args = ap.parse_args()

    refs = {label(args.teacher): load_adapter(args.teacher)}
    for o in args.others:
        refs[label(o)] = load_adapter(o)
    tname = label(args.teacher)

    rows = []

    def add(a_name, a, b_name, b):
        per, gcos, norm_a = compare(a, b)
        mean_c, med_c, by_type = summarize(per)
        rows.append(dict(a=a_name, b=b_name, global_cos=gcos, mean_mod_cos=mean_c,
                         median_mod_cos=med_c, norm_a=norm_a, by_type=by_type))

    # reference scale: teacher vs others, and others vs each other
    names = list(refs)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            add(names[i], refs[names[i]], names[j], refs[names[j]])
    for s in args.students:
        sa = load_adapter(s)
        for rn, r in refs.items():
            add(label(s), sa, rn, r)

    norms = {}
    for rn, r in refs.items():
        norms[rn] = sum(inner(v, v) for v in r.values()) ** 0.5

    print(f"teacher = {tname}   ({len(refs[tname])} LoRA modules)")
    print("\n||dW||_F (global):")
    for rn, n in norms.items():
        print(f"  {rn:32s} {n:9.4f}")
    for r in rows:
        if r["a"] not in norms:
            norms[r["a"]] = r["norm_a"]
            print(f"  {r['a']:32s} {r['norm_a']:9.4f}")

    hdr = f"\n{'A':32s} {'B':28s} {'global_cos':>10s} {'mean_mod':>9s} {'med_mod':>9s}"
    print(hdr)
    print("-" * (len(hdr) - 1))
    for r in rows:
        mark = " *" if r["b"] == tname else ""
        print(f"{r['a']:32s} {r['b']:28s} {r['global_cos']:10.4f} {r['mean_mod_cos']:9.4f} "
              f"{r['median_mod_cos']:9.4f}{mark}")

    if args.per_module:
        types = list(rows[0]["by_type"])
        print("\nper-module-type cosine (pooled within type):")
        print(f"{'A':32s} {'B':28s} " + " ".join(f"{t.split('.')[-1][:10]:>10s}" for t in types))
        for r in rows:
            print(f"{r['a']:32s} {r['b']:28s} " + " ".join(f"{r['by_type'][t]:10.4f}" for t in types))

    if args.csv:
        types = list(rows[0]["by_type"])
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["a", "b", "global_cos", "mean_mod_cos", "median_mod_cos", "norm_a"] + types)
            for r in rows:
                w.writerow([r["a"], r["b"], r["global_cos"], r["mean_mod_cos"], r["median_mod_cos"],
                            r["norm_a"]] + [r["by_type"][t] for t in types])
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
