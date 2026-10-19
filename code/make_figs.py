"""Figures for the proximity-coupling draft. Numbers copied from the result JSONs on jeff's PC
(dd_batch_diag*.json, dd_explore_2023h2_2025.json, dd_confirm_2026.json), relayed 2026-10-15.
v2 (2026-10-17): values from the corrected server key (cluster_id, vault_id, pod_id), *_rekeyc runs."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
CE, CC = "#1f77b4", "#d62728"
def eb(ax, x, v, c, lab, m="o"):
    v = np.array(v, float); ax.errorbar(x, v[:, 0], yerr=[v[:, 0]-v[:, 1], v[:, 2]-v[:, 0]], fmt=m, color=c, capsize=3, label=lab)

# Fig 1: sets, primary window (rest CI not relayed -> point only)
f, ax = plt.subplots(figsize=(3.6, 2.8)); x = np.arange(3)
eb(ax, x-0.1, [[1.231,1.156,1.325],[1.144,1.046,1.248],[0.984,0.955,1.017]], CE, "Exploration 2023H2–2025")
eb(ax, x+0.1, [[1.267,1.103,1.444],[0.953,0.787,1.120],[0.986,0.925,1.037]], CC, "Confirmation 2026H1", "s")
ax.axhline(1, color="grey", lw=0.8, ls="--"); ax.set_xticks(x, ["Same pod", "Same tome\n(stripe)", "Rest of vault"])
ax.set_ylabel("Excess failure risk X (days 1–30)"); ax.legend(frameon=False, fontsize=7); f.tight_layout(); f.savefig("fig/fig1_sets.png", dpi=300)

# Fig 2: distance bins
f, ax = plt.subplots(figsize=(3.6, 2.8)); x = np.arange(4)
eb(ax, x-0.1, [[2.025,1.732,2.331],[1.343,1.149,1.557],[1.187,1.077,1.313],[1.063,0.970,1.165]], CE, "Exploration")
eb(ax, x+0.1, [[2.279,1.709,2.853],[1.061,0.691,1.548],[1.215,0.971,1.476],[1.175,0.966,1.365]], CC, "Confirmation 2026H1", "s")
ax.axhline(1, color="grey", lw=0.8, ls="--"); ax.set_xticks(x, ["1–2", "3–5", "6–15", "≥16"]); ax.set_xlabel("Slot distance to failed drive")
ax.set_ylabel("Excess failure risk X (days 1–30)"); ax.legend(frameon=False, fontsize=7); f.tight_layout(); f.savefig("fig/fig2_distance.png", dpi=300)

# Fig 3: time course, all pod-mates
f, ax = plt.subplots(figsize=(3.6, 2.8)); x = np.arange(5); lab = ["1–3", "4–7", "8–14", "15–30", "31–60"]
pod_e = [[1.434,1.173,1.747],[1.416,1.234,1.602],[1.274,1.146,1.417],[1.110,1.022,1.203],[1.101,1.043,1.165]]
pod_c = [[2.184,1.703,2.765],[1.490,1.120,1.892],[0.949,0.723,1.175],[1.067,0.850,1.297],[1.033,0.866,1.183]]
eb(ax, x-0.1, pod_e, CE, "Exploration"); eb(ax, x+0.1, pod_c, CC, "Confirmation 2026H1", "s")
ax.axhline(1, color="grey", lw=0.8, ls="--"); ax.set_xticks(x, lab); ax.set_xlabel("Days after index failure")
ax.set_ylabel("Excess failure risk X"); ax.legend(frameon=False, fontsize=7); f.tight_layout(); f.savefig("fig/fig3_timecourse.png", dpi=300)

# Fig 4: near/far ratio by batch
f, ax = plt.subplots(figsize=(3.6, 2.8)); x = np.arange(3)
eb(ax, x-0.1, [[1.906,1.603,2.271],[1.832,1.533,2.166],[2.476,1.393,3.747]], CE, "Exploration")
eb(ax, x+0.1, [[1.939,1.405,2.609],[2.216,1.619,2.949],[0.793,0.220,1.761]], CC, "Confirmation 2026H1", "s")
ax.axhline(1, color="grey", lw=0.8, ls="--"); ax.set_xticks(x, ["All pairs", "Same batch", "Different batch"])
ax.set_ylabel("Near/far ratio (distance 1–2 vs ≥16)"); ax.legend(frameon=False, fontsize=7); f.tight_layout(); f.savefig("fig/fig4_batch.png", dpi=300)
