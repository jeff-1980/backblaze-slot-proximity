"""Schematic of the server structure and the near-to-far ratio (v10, 2026-10-19, ninth review R2-m3/R3-m2).
Panel a: a vault drawn as pods (rows) of numbered slots; one failed drive, its pod-mates at slot distance 1-2 (near)
and >=16 (far), and its tome-mates (same slot number in the other pods). Slot numbers are labels, not verified
geometry. Panel b: worked example of the near-to-far ratio from the exploration rows of Table 2 (O, E, placebo mean).
USAGE  python make_fig_schematic_v10.py <out.png>"""
import sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
plt.rcParams.update({"savefig.facecolor": "white", "font.size": 8})
CF, CN, CFAR, CT, CO = "#d62728", "#ff9f40", "#1f77b4", "#2ca02c", "#e6e6e6"
NS, NP, FS, FP = 30, 5, 10, 2          # slots drawn, pods drawn, failed slot, failed pod

f = plt.figure(figsize=(7.2, 3.6))
ax = f.add_axes([0.02, 0.08, 0.60, 0.80]); ax.set_xlim(-3.2, NS + 0.5); ax.set_ylim(-2.5, NP + 1.4); ax.axis("off")
for p in range(NP):
    y = NP - 1 - p
    ax.text(-0.4, y + 0.4, f"pod {p if p < NP - 1 else 19}", ha="right", va="center", fontsize=7)
    for s in range(NS):
        d = abs(s - FS)
        c = CO
        if p == FP:
            c = CF if s == FS else CN if 1 <= d <= 2 else CFAR if d >= 16 else "#bdbdbd"
        elif s == FS:
            c = CT
        ax.add_patch(Rectangle((s, y), 0.85, 0.8, color=c, lw=0))
    if p == NP - 2:
        ax.text(-1.6, y - 0.1, "⋮", ha="center", va="center", fontsize=9)
for s in (0, FS, 20, NS - 1):
    ax.text(s + 0.42, NP + 0.1, str(s), ha="center", va="bottom", fontsize=6.5)
ax.text(-3.2, NP + 0.55, "a", fontsize=11, fontweight="bold", va="bottom")
ax.text(1, NP + 0.55, "One vault: 20 pods (servers); slots 0–29 of a 60-slot pod shown.\nSlot numbers are labels, not verified physical geometry.", fontsize=7.5, va="bottom")
items = [(CF, "failed drive (index failure)"), (CN, "pod-mates at slot distance 1–2 (near)"),
         (CFAR, "pod-mates at distance ≥16 (far)"), ("#bdbdbd", "other pod-mates"), (CT, "tome-mates (same slot, other pods)")]
for i, (c, t) in enumerate(items):
    x0 = (i % 2) * 15; y0 = -0.75 - 0.7 * (i // 2)
    ax.add_patch(Rectangle((x0, y0), 0.85, 0.5, color=c, lw=0)); ax.text(x0 + 1.2, y0 + 0.25, t, va="center", fontsize=6.8)

bx = f.add_axes([0.64, 0.08, 0.35, 0.80]); bx.axis("off"); bx.set_xlim(0, 1); bx.set_ylim(0, 1)
bx.text(-0.02, 0.985, "b", fontsize=11, fontweight="bold", va="top")
lines = [("Worked example (exploration, days 1–30)", True),
         ("X = (O / E) / placebo-mean O/E", False),
         ("Near:  (245 / 106.4) / 1.133 = 2.03", False),
         ("Far:    (898 / 782.0) / 1.080 = 1.06", False),
         ("Near-to-far ratio = 2.03 / 1.06 ≈ 1.91", True),
         ("", False),
         ("O: recorded follow-on failures in the set", False),
         ("E: expected from model × age × vault", False),
         ("     baseline hazard and exposure days", False),
         ("Placebo: (pod, slot) labels permuted", False),
         ("     within vault × model × age bin", False),
         ("A ratio of event-clustering indices,", False),
         ("not a ratio of per-drive probabilities.", False)]
for i, (t, b) in enumerate(lines):
    bx.text(0.04, 0.88 - i * 0.068, t, fontsize=7.4, fontweight="bold" if b else "normal", va="top")
f.savefig(sys.argv[1], dpi=300, facecolor="white"); plt.close(f)
