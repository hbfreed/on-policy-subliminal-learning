import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
names = ["base"] + sorted((p.name for p in root.iterdir() if p.name.startswith("ep")),
                          key=lambda n: int(n[2:]))
MASC = ["MASC_trained", "MASC_heldout_names", "MASC_heldout_tmpl"]
print(f"{'':6s} {'FEM_tr':>7s} {'FEM_hn':>7s} {'FEM_ht':>7s} {'MASC':>6s} {'REF':>6s} "
      f"{'margin FEM-MASC':>16s}")
for n in names:
    p = root / n / "summary.json"
    if not p.exists():
        continue
    s = json.loads(p.read_text())

    def fr(c):
        return s[c]["french_rate"]
    masc = sum(fr(c) for c in MASC) / 3
    print(f"{n:6s} {fr('FEM_trained'):7.1%} {fr('FEM_heldout_names'):7.1%} "
          f"{fr('FEM_heldout_tmpl'):7.1%} {masc:6.1%} {fr('REF_none'):6.1%} "
          f"{s['_headline']['FEM_minus_MASC']:+16.2f}")
