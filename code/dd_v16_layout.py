"""Mechanism test L (server layout), EXPLORATORY. Written 2026-10-20 before running; criteria below fixed before any
estimate was seen. A two-draw test run (NPL=2) is inspected for errors only.

Question. Is the near-slot excess organised by the backplane boards that group drives, or by physical adjacency?
Layout assumption (stated, not verified from the data): in 60- and 45-slot servers (Backblaze Storage Pod design:
rows of 15 drives; Pod 6.0 has 12 backplanes for 60 drives, i.e. 5 drives per backplane) the 0-based slot number runs
along rows of 15 and along backplanes of 5, so block5 = slot // 5 is the backplane and row15 = slot // 15 the row.
If the numbering does not follow the hardware this way, a null result says nothing about backplanes.

Base: clock F of dd_v10_clock.py via dd_v15_review.build (same exposure, hazard, strata, placebo), window 1-30 days
(also 1-7 days), index failures whose index server is in the 60- or 45-slot capacity class (class under the placebo =
server the index drive is placed in). Sets of pod-mates of an index drive at slot s:
  SB1..SB4  exact slot distance k = 1..4, same block5            (same backplane)
  XB1..XB4  exact distance k = 1..4, other block5, same row15     (neighbouring backplane, same row)
  XR1       distance 1, other row15                               (row boundary: slots 14|15, 29|30, 44|45)
  COL       distance 15                                           (same position, adjacent row)
  ROW5      distance 5..14, same row15
  FAR       distance >= 16
X = (O/E) / placebo mean of O/E per set (NPL draws, SEED 20261020); E* = E x placebo mean.
Primary statistic BP (backplane contrast) = Mantel-Haenszel ratio over k = 1..4 of same-block versus other-block,
same-row sets:  BP = sum_k O_SB(k) E*_XB(k) / T_k  /  sum_k O_XB(k) E*_SB(k) / T_k,  T_k = E*_SB(k) + E*_XB(k).
Secondary: RW = X(XR1) / X(SB1 + XB1)  (row boundary versus same-row distance 1), COL and ROW5 X values.
Interval: joint vault bootstrap (B = 1000) across all four periods (pooled; also explore / hold-out separately);
placebo means held fixed.
Pre-set reading (pooled, window 1-30):
  "backplane-organised"  BP > 1 with 95% interval lower bound > 1;
  "not backplane-organised" BP interval includes 1 and upper bound < 1.5 (rules out a 1.5-fold same-board effect);
  otherwise "inconclusive". Same rule reported for cluster 0 only and per period group (descriptive).
  RW, COL, ROW5 are descriptive (few events at row boundaries).
USAGE  NPL=100 python dd_v16_layout.py <out.json> <runs_dir> <k2023h2c> <k2024> <k2025> <k2026c>
"""
import sys, os, json, importlib.util, time
import numpy as np
import multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("v15", os.path.join(HERE, "dd_v15_review.py"))
v15 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v15)

NPL = int(os.environ.get("NPL", 100)); B = 1000; SEED = 20261020
WA = np.array([1, 1]); WB = np.array([30, 7]); WN = ["w1_30", "w1_7"]
NAMES = [f"SB{k}" for k in range(1, 5)] + [f"XB{k}" for k in range(1, 5)] + ["XR1", "COL", "ROW5", "FAR"]
NS = len(NAMES)
LAB = ["2023H2", "2024", "2025", "2026H1"]
GROUPS = {"pooled": [0, 1, 2, 3], "explore": [0, 1, 2], "confirm": [3]}
DATA = None
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def masks(s0, so, dist):
    b0, bo, r0, ro = s0 // 5, so // 5, s0 // 15, so // 15
    m = []
    for k in range(1, 5):
        m.append((dist == k) & (bo == b0))
    for k in range(1, 5):
        m.append((dist == k) & (bo != b0) & (ro == r0))
    m.append((dist == 1) & (ro != r0))
    m.append(dist == 15)
    m.append((dist >= 5) & (dist <= 14) & (ro == r0))
    m.append(dist >= 16)
    return np.stack(m).astype(float)


def tally(P, pod, slot):
    mf, ml, mfd, h = P["base"]
    ifd = P["F"]; vid = P["vid"]
    key = vid.astype(np.int64) * 1000 + pod.astype(np.int64)
    order = np.argsort(key, kind="stable"); uk, start, cnt = np.unique(key[order], return_index=True, return_counts=True)
    gpos = np.searchsorted(uk, key)
    idx = np.where(~np.isnan(ifd))[0]
    cm = P["cls_map"]
    cls = np.array([cm[(P["vault"][j], pod[j])] for j in idx], int)
    OE = np.zeros((len(idx), 2, NS, 2))
    for n_, j in enumerate(idx):
        if cls[n_] > 1:
            continue
        g = gpos[j]; mem = order[start[g]:start[g] + cnt[g]]; mem = mem[mem != j]
        t = ifd[j]
        M = masks(slot[j], slot[mem], np.abs(slot[mem] - slot[j]))
        f_, l_, fd_, h_ = mf[mem], ml[mem], mfd[mem], h[mem]
        st = np.maximum(f_[None, :], t + WA[:, None]); en = np.minimum(l_[None, :], t + WB[:, None])
        DD = np.clip(en - st + 1, 0, None)
        EV = ((fd_[None, :] >= t + WA[:, None]) & (fd_[None, :] <= t + WB[:, None])).astype(float)
        OE[n_, 0] = M @ EV.T
        OE[n_, 1] = (M * h_[None, :]) @ DD.T
    return {"OE": OE, "vix": vid[idx], "cls": cls, "ccl": P["vcl"][vid[idx]]}


def one(i):
    rng = np.random.default_rng([SEED, i]); out = []
    for P in DATA:
        pod, slot = v15.perm(P, P["strata_base"], rng)
        T = tally(P, pod, slot)
        m0 = T["ccl"] == 0
        out.append((T["OE"].sum(0).astype(np.float32), T["OE"][m0].sum(0).astype(np.float32)))
    return i, out


def pct(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))) if len(x) else (None, None)


def verdict(r, lo, hi):
    if lo is None:
        return "not_estimable"
    if lo > 1:
        return "backplane-organised"
    if lo <= 1 <= hi and hi < 1.5:
        return "not backplane-organised"
    return "inconclusive"


def main(outp, runs_dir, paths):
    global DATA
    D = [v15.build(p, runs_dir) for p in paths]
    for P in D:
        log("built", P["label"], P["n"], "drives", int((~np.isnan(P["F"])).sum()), "failures", P["server_classes"])
    OBS = [tally(P, P["pod"], P["slot"]) for P in D]
    log("observed tally done")
    DATA = D
    PL = [[None] * NPL for _ in D]
    with mp.get_context("fork").Pool(4) as pool:
        for k, (i, out) in enumerate(pool.imap_unordered(one, range(NPL), chunksize=1)):
            for p in range(len(D)):
                PL[p][i] = out[p]
            if k % 10 == 0 or k == NPL - 1:
                log("placebo", k + 1, "of", NPL)
    PLs = [(np.stack([x[0] for x in PL[p]]), np.stack([x[1] for x in PL[p]])) for p in range(len(D))]
    U = np.unique(np.concatenate([P["uv"] for P in D])); uidx = [np.searchsorted(U, P["uv"]) for P in D]
    rng = np.random.default_rng(SEED)
    mult = np.stack([np.bincount(rng.integers(0, len(U), len(U)), minlength=len(U)) for _ in range(B)]).astype(float)

    def group(ps, w, c0):
        k = 1 if c0 else 0
        po = sum(PLs[p][k][:, 0, :, w] for p in ps); pe = sum(PLs[p][k][:, 1, :, w] for p in ps)
        pm = (po / np.maximum(pe, 1e-12)).mean(0)                          # (NS,)
        vo = np.zeros((len(U), NS)); ve = np.zeros((len(U), NS))
        for p in ps:
            T = OBS[p]; m = (T["cls"] <= 1) & ((T["ccl"] == 0) if c0 else True)
            for s in range(NS):
                np.add.at(vo[:, s], uidx[p][T["vix"][m]], T["OE"][m, 0, s, w])
                np.add.at(ve[:, s], uidx[p][T["vix"][m]], T["OE"][m, 1, s, w])
        O, E = vo.sum(0), ve.sum(0); bo, be = mult @ vo, mult @ ve
        X = O / np.maximum(E, 1e-12) / pm; Xb = bo / np.maximum(be, 1e-12) / pm
        res = {"sets": {}}
        for s, nm in enumerate(NAMES):
            lo, hi = pct(Xb[:, s])
            res["sets"][nm] = {"O": int(round(O[s])), "E": float(E[s]), "placebo_mean": float(pm[s]), "X": float(X[s]), "lo": lo, "hi": hi}

        def mh(o, e):  # o, e: (..., NS) arrays; returns BP
            es = e * pm
            num = sum(o[..., k] * es[..., 4 + k] / np.maximum(es[..., k] + es[..., 4 + k], 1e-12) for k in range(4))
            den = sum(o[..., 4 + k] * es[..., k] / np.maximum(es[..., k] + es[..., 4 + k], 1e-12) for k in range(4))
            return num / np.maximum(den, 1e-12)
        bp = float(mh(O, E)); lo, hi = pct(mh(bo, be))
        res["BP"] = {"ratio": bp, "lo": lo, "hi": hi, "O_same_block_k1_4": int(round(O[:4].sum())),
                     "O_other_block_k1_4": int(round(O[4:8].sum())), "verdict": verdict(bp, lo, hi)}
        d1 = lambda o, e: (o[..., 0] + o[..., 4]) / np.maximum(e[..., 0] * pm[0] + e[..., 4] * pm[4], 1e-12)
        rw = lambda o, e: (o[..., 8] / np.maximum(e[..., 8] * pm[8], 1e-12)) / np.maximum(d1(o, e), 1e-12)
        lo, hi = pct(rw(bo, be))
        res["RW"] = {"ratio": float(rw(O, E)), "lo": lo, "hi": hi, "O_row_boundary_d1": int(round(O[8]))}
        return res

    res = {"meta": {"script": "code/dd_v16_layout.py", "status": "exploratory mechanism test, written before running",
                    "NPL": NPL, "B": B, "SEED": SEED, "periods": [P["label"] for P in D],
                    "layout_assumption": "0-based slot runs along rows of 15 and backplanes of 5 (Storage Pod design); unverified",
                    "server_classes": {P["label"]: P["server_classes"] for P in D}}}
    for g, ps in GROUPS.items():
        for w, wn in enumerate(WN):
            for c0 in (False, True):
                key = f"{g}|{wn}|{'cluster0' if c0 else 'all_45_60'}"
                res[key] = group(ps, w, c0)
                log(key, "BP", round(res[key]["BP"]["ratio"], 3), res[key]["BP"]["lo"], res[key]["BP"]["hi"], res[key]["BP"]["verdict"])
    res["primary"] = res["pooled|w1_30|all_45_60"]["BP"]
    os.makedirs(os.path.dirname(outp) or ".", exist_ok=True)
    json.dump(res, open(outp, "w"), indent=1, default=float)
    log("wrote", outp)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:7])
