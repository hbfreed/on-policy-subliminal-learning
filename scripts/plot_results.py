"""Plot teacher / student eval curves for the writeup.

Reads runs/<name>_eval/<epoch>/summary.json (epoch dirs: base, ep1, ep2, ...).
Also looks under runs/verda_backup/ for arms mirrored from the cloud box.
Missing dirs / epochs are skipped.

Usage:
    scripts/fetch_evals.sh /path/to/evals
    python scripts/plot_results.py --runs /path/to/evals --out /path/to/figs
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Categorical palette, fixed order (dataviz default instance).
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
GRAY = "#8a8985"
INK, INK2 = "#1f1f1e", "#5f5e5a"

SFT_TEACHERS = [  # (eval dir, label, color in margin panel)
    ("teacher_sft_s1_eval", "SFT seed 1", AQUA),
    ("teacher_sft_s2_eval", "SFT seed 2", YELLOW),
    ("teacher_sft_s1223_eval", "SFT seed 1223", MAGENTA),
]
OPSD_TEACHER = ("teacher_opsd_eval", "OPSD teacher")
STUDENTS = [
    ("student_offpolicy_s1_eval", "Off-policy (fwd KL)", BLUE),
    ("student_onpolicy_rev_eval", "On-policy (rev KL)", ORANGE),
    ("student_onpolicy_fwd_eval", "On-policy (fwd KL)", AQUA),
]

# Paper Table 4, students at epoch 16 (three seeds).
PAPER_EPOCH = 16
PAPER_FEM_FRENCH = [15.0, 16.8, 38.8]  # %, trained female names
PAPER_GAP = [10.0, 10.5, 15.6]  # nats
PAPER_TEACHER_GAP = [35.3, 41.5, 49.5]  # nats


def style():
    plt.rcParams.update({
        "figure.dpi": 100,
        "savefig.dpi": 200,
        "font.size": 10,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 9.5,
        "axes.labelcolor": INK2,
        "axes.edgecolor": "#c9c8c2",
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": "#e6e5e0",
        "grid.linewidth": 0.7,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "legend.frameon": False,
        "lines.linewidth": 2,
        "lines.markersize": 4.5,
        "text.color": INK,
    })


def find_eval_dir(runs: Path, name: str):
    for cand in (runs / name, runs / "verda_backup" / name, runs / "verda_backup" / "runs" / name):
        if cand.is_dir():
            return cand
    hits = sorted((runs / "verda_backup").glob(f"**/{name}")) if (runs / "verda_backup").is_dir() else []
    return hits[0] if hits else None


def load_run(runs: Path, name: str):
    """Return sorted list of (epoch_int, summary_dict); base -> 0. Empty if missing."""
    d = find_eval_dir(runs, name)
    if d is None:
        return []
    out = []
    for sub in d.iterdir():
        m = re.fullmatch(r"base|ep(\d+)", sub.name)
        f = sub / "summary.json"
        if not m or not f.is_file():
            continue
        try:
            s = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        out.append((0 if sub.name == "base" else int(m.group(1)), s))
    return sorted(out, key=lambda t: t[0])


def series(rows, cond, key, scale=1.0):
    xs, ys = [], []
    for ep, s in rows:
        if cond in s and key in s[cond]:
            xs.append(ep)
            ys.append(s[cond][key] * scale)
    return xs, ys


def save(fig, out: Path, stem: str):
    png = out / f"{stem}.png"
    fig.savefig(png, bbox_inches="tight", facecolor="white")
    written = [png]
    try:
        from PIL import Image

        webp = out / f"{stem}.webp"
        Image.open(png).save(webp, "WEBP", quality=90, method=6)
        written.append(webp)
    except ImportError:
        pass
    plt.close(fig)
    for p in written:
        print("wrote", p)


def plot_teachers(runs: Path, out: Path):
    teachers = [(lbl, c, load_run(runs, n)) for n, lbl, c in SFT_TEACHERS]
    teachers = [t for t in teachers if t[2]]
    opsd = load_run(runs, OPSD_TEACHER[0])
    if not teachers and not opsd:
        print("skip teachers_sft: no data")
        return

    n = max(len(teachers), 1)
    fig = plt.figure(figsize=(8, 6.4))
    gs = fig.add_gridspec(2, n, height_ratios=[1, 1.25], hspace=0.55, wspace=0.12)

    max_ep = max([r[-1][0] for _, _, r in teachers] + ([opsd[-1][0]] if opsd else []))
    xt = list(range(0, max_ep + 1, 2 if max_ep > 8 else 1))

    for i, (lbl, _, rows) in enumerate(teachers):
        ax = fig.add_subplot(gs[0, i])
        for cond, c, name in (("FEM_trained", BLUE, "female names"), ("MASC_trained", ORANGE, "male names")):
            x, y = series(rows, cond, "french_rate", 100)
            ax.plot(x, y, color=c, marker="o", label=name)
        x, y = series(rows, "REF_none", "french_rate", 100)
        ax.plot(x, y, color=GRAY, lw=1.2, ls=":", label="no name")
        ax.set_title(lbl)
        ax.set_ylim(-3, 103)
        ax.set_xticks(xt)
        ax.set_xlabel("teacher epoch")
        if i == 0:
            ax.set_ylabel("French responses (%)")
        else:
            ax.tick_params(labelleft=False)
        if i == n - 1:
            ax.legend(loc="center right")

    ax = fig.add_subplot(gs[1, :])
    for lbl, c, rows in teachers:
        x, y = [e for e, _ in rows], [s["_headline"]["FEM_minus_MASC"] for _, s in rows]
        ax.plot(x, y, color=c, marker="o", label=lbl)
    if opsd:
        x, y = [e for e, _ in opsd], [s["_headline"]["FEM_minus_MASC"] for _, s in opsd]
        ax.plot(x, y, color=GRAY, ls="--", marker="o", mfc="white", label=f"{OPSD_TEACHER[1]} (0% French)")
    # Paper teacher gaps (final teachers) as reference ticks on the right edge.
    xr = max_ep + 0.6
    ax.scatter([xr] * len(PAPER_TEACHER_GAP), PAPER_TEACHER_GAP, marker="_", s=120, color=INK2, lw=2,
               zorder=3, clip_on=False, label="paper teachers (3 seeds)")
    ax.axhline(0, color="#c9c8c2", lw=0.8, zorder=0)
    ax.set_xlim(-0.4, max_ep + 1.2)
    ax.set_xticks(xt)
    ax.set_xlabel("teacher epoch")
    ax.set_ylabel("female − male margin (nats)")
    ax.set_title("Gender-conditional margin gap (log P(French) − log P(English), pooled)")
    ax.legend(loc="upper left", ncol=1)

    fig.suptitle("SFT teachers: any name → French first, then gender-specific", x=0.065, ha="left",
                 fontsize=12, fontweight="bold", y=0.995)
    save(fig, out, "teachers_sft")


def plot_students(runs: Path, out: Path):
    arms = [(lbl, c, load_run(runs, n)) for n, lbl, c in STUDENTS]
    present = [a for a in arms if a[2]]
    if not present:
        print("skip students: no data")
        return

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8, 3.6), gridspec_kw={"wspace": 0.28})
    max_ep = max(PAPER_EPOCH, max(r[-1][0] for _, _, r in present))

    for lbl, c, rows in present:
        x, y = series(rows, "FEM_trained", "french_rate", 100)
        a1.plot(x, y, color=c, marker="o", label=lbl)
        x, y = [e for e, _ in rows], [s["_headline"]["FEM_minus_MASC"] for _, s in rows]
        a2.plot(x, y, color=c, marker="o", label=lbl)

    for ax, vals in ((a1, PAPER_FEM_FRENCH), (a2, PAPER_GAP)):
        lo, hi, mean = min(vals), max(vals), sum(vals) / len(vals)
        ax.fill_between([PAPER_EPOCH - 0.6, PAPER_EPOCH + 0.6], lo, hi, color=GRAY, alpha=0.18, lw=0, zorder=0)
        ax.scatter([PAPER_EPOCH] * len(vals), vals, marker="D", s=22, color=INK2, zorder=3,
                   label="paper, epoch 16 (3 seeds; band = range, bar = mean)")
        ax.scatter([PAPER_EPOCH], [mean], marker="_", s=180, color=INK, lw=2, zorder=3)
        ax.axhline(0, color="#c9c8c2", lw=0.8, zorder=0)
        ax.set_xlim(-0.5, max_ep + 0.8)
        ax.set_xticks(range(0, max_ep + 1, 2))
        ax.set_xlabel("student epoch")

    a1.text(PAPER_EPOCH - 0.8, sum(PAPER_FEM_FRENCH) / 3, "paper\nmean", ha="right", va="center", fontsize=8, color=INK2)
    a1.set_ylim(-2, max(45, a1.get_ylim()[1]))
    a1.set_ylabel("French responses (%)")
    a1.set_title("French rate, trained female names")
    a2.set_ylabel("female − male margin (nats)")
    a2.set_title("Margin gap (female − male)")

    handles, labels = a2.get_legend_handles_labels()
    missing = [lbl for lbl, _, rows in arms if not rows]
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), bbox_to_anchor=(0.5, -0.1))
    title = "Students: subliminal transfer of the female → French preference"
    if missing:
        fig.text(0.99, -0.16, "not yet available: " + ", ".join(missing), ha="right", fontsize=7.5, color=INK2)
    fig.suptitle(title, x=0.065, ha="left", fontsize=12, fontweight="bold", y=1.02)
    save(fig, out, "students")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, required=True, help="local copy of runs/ (see scripts/fetch_evals.sh)")
    ap.add_argument("--out", type=Path, required=True, help="output directory for figures")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    style()
    plot_teachers(args.runs, args.out)
    plot_students(args.runs, args.out)


if __name__ == "__main__":
    main()
