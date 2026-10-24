"""Within-server slot changes, sensitivity of R (mock review 2026-10-24, R3-M2). EXPLORATORY. Written before running.

Question. The main analysis gives each drive its last recorded slot in a period. Drives that changed slot within the
same server during the period could place an event at the wrong distance. How many such drives are there, how many
near events involve them, and does R change when they are removed?

Slot changers: serials whose daily location runs (results/v6runs, dd_v6_runs.py) show more than one slot number at the
same (cluster, vault, pod) within a period.
Base: clock F via dd_v15_review.build (same exposure, hazard, strata and main placebo), window days 1-30, all servers.
Variants: "all" = every drive (reproduces the clock-F run of dd_v15_review up to placebo draws);
          "no_changers" = slot changers removed as index failures and as members of the near and far sets, in the
          observed tally and in every placebo draw (baseline hazards unchanged).
Sets: NEAR = slot distance 1-2, FAR = distance >= 16. X = (O/E)/placebo mean, R = X_NEAR/X_FAR; NPL placebo draws
(SEED 20261024), vault bootstrap B = 1000 across all four periods, placebo means fixed; groups pooled, explore, confirm.
Also reported: number of slot changers per period and observed near events with a changer as index or follow-on drive.
No reading rule beyond reporting: the result is a sensitivity analysis of the main estimate.
USAGE  NPL=300 python dd_v17_slotchange.py <out.json> <runs_dir> <k2023h2c> <k2024> <k2025> <k2026c>
"""
import sys, os, json, importlib.util, time
import numpy as np
import pandas as pd
import multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("v15", os.path.join(HERE, "dd_v15_review.py"))
v15 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v15)
spec = importlib.util.spec_from_file_location("v10", os.path.join(HERE, "dd_v10_clock.py"))
v10 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v10)

NPL = int(os.environ.get("NPL", 300)); B = 1000; SEED = 20261024
W = (1, 30)
GROUPS = {"pooled": [0, 1, 2, 3], "explore": [0, 1, 2], "confirm": [3]}
VARS = ["all", "no_changers"]
DATA = None
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def changers(runs_path):
    r = pd.read_parquet(runs_path, columns=["s", "c", "v", "p", "sl"]).dropna(subset=["sl"])
    g = r.groupby(["s", "c", "v", "p"]).sl.nunique()
    return set(g[g > 1].reset_index().s.unique())


def tally(P, pod, slot):
    f_all, l_all, fd_all, h_all = P["base"]
    ifd = P["F"]; vid = P["vid"]; chg = P["chg"]
    key = vid.astype(np.int64) * 1000 + pod.astype(np.int64)
    order = np.argsort(key, kind="stable"); uk, start, cnt = np.unique(key[order], return_index=True, return_counts=True)
    gpos = np.searchsorted(uk, key)
    idx = np.where(~np.isnan(ifd))[0]
    OE = np.zeros((len(idx), 2, 2, 2))      # index, variant, set (near, far), (O, E)
    for n_, j in enumerate(idx):
        g = gpos[j]; mem = order[start[g]:start[g] + cnt[g]]; mem = mem[mem != j]
        t = ifd[j]
        dist = np.abs(slot[mem] - slot[j])
        M = np.stack([(dist >= 1) & (dist <= 2), dist >= 16]).astype(float)
        st = np.maximum(f_all[mem], t + W[0]); en = np.minimum(l_all[mem], t + W[1])
        DD = np.clip(en - st + 1, 0, None)
        EV = ((fd_all[mem] >= t + W[0]) & (fd_all[mem] <= t + W[1])).astype(float)
        o, e = M @ EV, M @ (h_all[mem] * DD)
        OE[n_, 0, :, 0], OE[n_, 0, :, 1] = o, e
        if not chg[j]:
            keep = (~chg[mem]).astype(float)
            OE[n_, 1, :, 0], OE[n_, 1, :, 1] = M @ (EV * keep), M @ (h_all[mem] * DD * keep)
    return {"OE": OE, "vix": vid[idx]}


def one(i):
    rng = np.random.default_rng([SEED, i]); out = []
    for P in DATA:
        pod, slot = v15.perm(P, P["strata_base"], rng)
        out.append(tally(P, pod, slot)["OE"].sum(0).astype(np.float32))
    return i, out


def pct(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))) if len(x) else (None, None)


def main(outp, runs_dir, paths):
    global DATA
    D = []
    meta = {}
    for p in paths:
        P = v15.build(p, runs_dir)
        ch = changers(os.path.join(runs_dir, v10.RUNS[os.path.basename(p)]))
        P["chg"] = np.isin(P["serial"], list(ch))
        meta[P["label"]] = {"slot_changers_in_runs": len(ch), "slot_changers_in_table": int(P["chg"].sum()),
                            "index_failures": int((~np.isnan(P["F"])).sum()),
                            "index_failures_of_changers": int((P["chg"] & ~np.isnan(P["F"])).sum())}
        log("built", P["label"], meta[P["label"]])
        D.append(P)
    OBS = [tally(P, P["pod"], P["slot"]) for P in D]
    for P, T in zip(D, OBS):
        o = T["OE"].sum(0)
        meta[P["label"]].update(near_events_all=float(o[0, 0, 0]), near_events_no_changers=float(o[1, 0, 0]),
                                far_events_all=float(o[0, 1, 0]), far_events_no_changers=float(o[1, 1, 0]))
    log("observed", meta)
    DATA = D
    PL = [[None] * NPL for _ in D]
    with mp.get_context("fork").Pool(4) as pool:
        for k, (i, out) in enumerate(pool.imap_unordered(one, range(NPL), chunksize=1)):
            for p in range(len(D)):
                PL[p][i] = out[p]
            if k % 20 == 0 or k == NPL - 1:
                log("placebo", k + 1, "of", NPL)
    PL = [np.stack(x) for x in PL]                      # (NPL, variant, set, O/E)
    U = np.unique(np.concatenate([P["uv"] for P in D])); uidx = [np.searchsorted(U, P["uv"]) for P in D]
    rng = np.random.default_rng(SEED)
    mult = np.stack([np.bincount(rng.integers(0, len(U), len(U)), minlength=len(U)) for _ in range(B)]).astype(float)
    res = {"meta": {"script": "code/dd_v17_slotchange.py", "status": "exploratory sensitivity, written before running",
                    "NPL": NPL, "B": B, "SEED": SEED, "window": W, "periods": [P["label"] for P in D], "per_period": meta}}
    for g, ps in GROUPS.items():
        for vi, vn in enumerate(VARS):
            po = sum(PL[p][:, vi, :, 0] for p in ps); pe = sum(PL[p][:, vi, :, 1] for p in ps)
            pm = (po / np.maximum(pe, 1e-12)).mean(0)
            vo = np.zeros((len(U), 2)); ve = np.zeros((len(U), 2))
            for p in ps:
                T = OBS[p]
                for s in range(2):
                    np.add.at(vo[:, s], uidx[p][T["vix"]], T["OE"][:, vi, s, 0])
                    np.add.at(ve[:, s], uidx[p][T["vix"]], T["OE"][:, vi, s, 1])
            O, E = vo.sum(0), ve.sum(0); bo, be = mult @ vo, mult @ ve
            X = O / E / pm; Xb = bo / np.maximum(be, 1e-12) / pm
            Rb = Xb[:, 0] / Xb[:, 1]; lo, hi = pct(Rb)
            res[f"{g}|{vn}"] = {"O_near": float(O[0]), "E_near": float(E[0]), "pm_near": float(pm[0]),
                                "O_far": float(O[1]), "E_far": float(E[1]), "pm_far": float(pm[1]),
                                "X_near": float(X[0]), "X_far": float(X[1]), "R": float(X[0] / X[1]), "lo": lo, "hi": hi}
            log(g, vn, round(X[0] / X[1], 3), lo, hi)
    json.dump(res, open(outp, "w"), indent=1, default=float)
    log("wrote", outp)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:7])
