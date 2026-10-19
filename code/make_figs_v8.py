"""Figures for v8 (2026-10-19, seventh review: larger canvas, one-line axis titles, boxed legend above the plot area so neither can be cropped) and v7 (2026-10-19, sixth review: wider padding and a white background so no axis title is cut when the PNG is placed in Word; otherwise as v5) and v5 (2026-10-18, fourth review: axis labels name X an event-clustering index; Fig. 5 x-label no longer cropped), read from the unified run (dd_unified_analyse.py output).
Fig. 4 different-batch values are not in the unified run and are taken from dd_batch_diag (frozen run).
USAGE  python code/make_figs_v8.py results/unified_rekeyc.json   (run from the repository root; writes figures/)"""
import json, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
plt.rcParams.update({"savefig.facecolor": "white", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
CE, CC = "#1f77b4", "#d62728"
R = json.load(open(sys.argv[1])); G = R["groups"]
T = lambda g, k: [G[g]["tables"][k][x] for x in ("X", "lo", "hi")]
Rt = lambda d: [d["ratio"], d["lo"], d["hi"]]


def eb(ax, x, v, c, lab, m="o"):
    v = np.array(v, float); ax.errorbar(x, v[:, 0], yerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt=m, color=c, capsize=3, label=lab)


def two(fn, keys, xt, xl, yl, getter):
    f, ax = plt.subplots(figsize=(4.6, 3.4)); x = np.arange(len(keys))
    eb(ax, x - 0.1, [getter("explore", k) for k in keys], CE, "Exploration 2023H2–2025")
    eb(ax, x + 0.1, [getter("confirm", k) for k in keys], CC, "Hold-out 2026H1", "s")
    ax.axhline(1, color="grey", lw=0.8, ls="--"); ax.set_xticks(x, xt)
    if xl:
        ax.set_xlabel(xl)
    ax.set_ylabel(yl); ax.legend(frameon=True, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2, edgecolor="0.7"); f.subplots_adjust(left=0.2, bottom=0.18, top=0.86, right=0.92); f.savefig(fn, dpi=300, facecolor="white"); plt.close(f)


two("figures/fig1_sets.png", ["POD|P", "TOME|P", "REST|P"], ["Same pod", "Same tome\n(stripe)", "Rest of vault"], None,
    "Excess X, days 1–30", T)
two("figures/fig2_distance.png", ["d1|P", "d2|P", "d3|P", "d4|P"], ["1–2", "3–5", "6–15", "≥16"], "Slot distance to failed drive",
    "Excess X, days 1–30", T)
two("figures/fig3_timecourse.png", [f"POD|L{i}" for i in range(1, 6)], ["1–3", "4–7", "8–14", "15–30", "31–60"], "Days after index failure",
    "Excess X (event-clustering index)", T)
ob = {"explore": [2.476, 1.393, 3.747], "confirm": [0.793, 0.220, 1.761]}
two("figures/fig4_batch.png", ["all", "sb", "ob"], ["All pairs", "Same batch", "Different batch"], None,
    "Near/far ratio (distance 1–2 vs ≥16)",
    lambda g, k: Rt(G[g]["hyp"]["H4_near_far"]) if k == "all" else Rt(G[g]["hyp"]["H7_samebatch"]) if k == "sb" else ob[g])
PS = R["pooled_subgroups"]; by = PS["by"]
rows = [("All periods pooled", PS["all"])]
rows += [("Period " + lb, by["period"][str(i)]) for i, lb in enumerate(("2023H2", "2024", "2025", "2026H1"))]
rows += [("Data centre " + k, v) for k, v in sorted(by["dc"].items()) if v["n_d1"] >= 10]
rows += [("Cluster 0", by["cluster0"]["c0"]), ("Other clusters pooled", by["cluster0"]["other"])]
f, ax = plt.subplots(figsize=(5.6, 4.2)); y = np.arange(len(rows))[::-1]
for yy, (n, v) in zip(y, rows):
    ax.errorbar(v["ratio"], yy, xerr=[[v["ratio"] - v["lo"]], [v["hi"] - v["ratio"]]], fmt="o", color=CE, capsize=2)
    ax.text(3.55, yy, f"{v['n_d1']}", va="center", fontsize=7)
ax.axvline(1, color="grey", lw=0.8, ls="--"); ax.set_yticks(y, [n for n, _ in rows]); ax.set_xlim(0, 3.8)
ax.text(3.55, len(rows) - 0.3, "n near", fontsize=7)
ax.set_xlabel("Near/far ratio\n(slot distance 1–2 vs ≥16, days 1–30)"); f.tight_layout()
f.savefig("figures/fig5_generalisation.png", dpi=300, bbox_inches="tight", pad_inches=0.25, facecolor="white")
print("figures written")
