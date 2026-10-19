"""Unified high-precision run (third review, 2026-10-18). Written before running; replaces the several
placebo runs (100 frozen, 50 audit, 200 review, 100 generalisability) as the single source of current
point estimates. The frozen outputs stay archived unchanged.

One tally per period with all sets and windows, on the corrected key (cluster, vault, pod).
Sets: POD, d0 (own slot), d1 1-2, d2 3-5, d3 6-15, d4 >=16, TOME, REST, d1/d4 same-batch with the frozen H7
definition (same model, |poh0 diff| < 720 h at first appearance).
Windows (days after the index day): P 1-30, L1 1-3, L2 4-7, L3 8-14, L4 15-30, L5 31-60, E7 1-7, LATE 15-60,
M 8-30, G1 61-120, G2 121-180 (long lags: a fixed high-risk slot region predicts a near excess at every lag;
a response to the failure predicts one only at short lags).
Placebos:
  main  (pod, slot) permuted jointly within vault x model x age bin (as frozen), NPL_MAIN draws;
  C1    slot labels permuted among all drives of the pod (as review 2), NPL_POD draws;
  C3    slot-map placebo: a random permutation of the slot numbers of each pod applied to every drive of
        the pod, so drives that held the same slot one after another move together and no two drives
        alive at the same time can share a slot, NPL_POD draws.
Also: C1 co-occupancy check (share of placebo slot assignments that put two drives with overlapping
residence in one slot), exchangeability counts, absolute risk per vault (index failures with a complete
30-day window) and the same with each exposed drive counted once per period.
Output: <out>.npz (per-vault arrays) and <out>_meta.json. Analysis in dd_unified_analyse.py.
USAGE  python dd_unified.py <out_prefix> <period.parquet> [...]
"""
import sys, os, json, importlib.util
import numpy as np
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("r2", os.path.join(HERE, "dd_review2.py"))
r2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(r2)

NPL_MAIN = int(os.environ.get("NPL_MAIN", 1000)); NPL_POD = int(os.environ.get("NPL_POD", 300))
SEED = 20261018
WINS = {"P": (1, 30), "L1": (1, 3), "L2": (4, 7), "L3": (8, 14), "L4": (15, 30), "L5": (31, 60),
        "E7": (1, 7), "LATE": (15, 60), "M": (8, 30), "G1": (61, 120), "G2": (121, 180)}
WL = list(WINS)
DB = [(0, 0, "d0"), (1, 2, "d1"), (3, 5, "d2"), (6, 15, "d3"), (16, 10 ** 6, "d4")]
SETS = ["POD", "d0", "d1", "d2", "d3", "d4", "TOME", "REST", "d1_sb", "d4_sb"]
ABS_SETS = ["POD", "d1", "d2", "d3", "d4", "REST"]
PL_KEEP = [("POD", "P"), ("d1", "P"), ("d4", "P"), ("TOME", "P"), ("REST", "P")]
PER = None


def tally(d, pod, slot, detail=False):
    uv = np.unique(d.vault.values); nv = len(uv)
    O = np.zeros((len(SETS), len(WL), nv)); E = np.zeros_like(O)
    A = {"exp": np.zeros((len(ABS_SETS), nv)), "ev": np.zeros((len(ABS_SETS), nv)), "eexp": np.zeros((len(ABS_SETS), nv))}
    dedup = {s: [set(), set()] for s in ABS_SETS}
    first, last, fd, h = d["first"].values, d["last"].values, d.fdate.values, d.h.values
    mdl, poh = d.model.values, d.poh0.values.astype(float)
    pend = d.pend.values[0]
    wa = np.array([WINS[w][0] for w in WL]); wb = np.array([WINS[w][1] for w in WL])
    vidx = {v: k for k, v in enumerate(uv)}
    for v, ix in d.groupby("vault").indices.items():
        k = vidx[v]
        f_, l_, fd_, h_, p_, s_, m_, a_ = first[ix], last[ix], fd[ix], h[ix], pod[ix], slot[ix], mdl[ix], poh[ix]
        n = len(ix)
        for j in np.where(~np.isnan(fd_))[0]:
            t = fd_[j]
            other = np.ones(n, bool); other[j] = False
            sp = p_ == p_[j]; ss = s_ == s_[j]; dist = np.abs(s_ - s_[j])
            pm = other & sp
            sb = (m_ == m_[j]) & (np.abs(a_ - a_[j]) < 720)
            M = [pm]
            for lo, hi, _ in DB:
                M.append(pm & (dist >= lo) & (dist <= hi))
            M += [other & ss & ~sp, other & ~sp & ~ss, M[2] & sb, M[5] & sb]
            M = np.array(M, dtype=float)                      # sets x n
            st = np.maximum(f_[None, :], t + wa[:, None]); en = np.minimum(l_[None, :], t + wb[:, None])
            DD = np.clip(en - st + 1, 0, None)                # wins x n
            EV = ((fd_[None, :] >= t + wa[:, None]) & (fd_[None, :] <= t + wb[:, None])).astype(float)
            O[:, :, k] += M @ EV.T
            E[:, :, k] += (M * h_[None, :]) @ DD.T
            if detail and t <= pend - 30:
                alive = (l_ >= t + 1) & ~(fd_ <= t)
                ev30 = EV[0] > 0
                for si, s in enumerate(ABS_SETS):
                    mm = M[SETS.index(s)] > 0
                    am = mm & alive
                    A["exp"][si, k] += am.sum(); A["ev"][si, k] += (am & ev30).sum()
                    A["eexp"][si, k] += (h_[am] * DD[0][am]).sum()
                    dedup[s][0].update(ix[am].tolist()); dedup[s][1].update(ix[am & ev30].tolist())
    if detail:
        A["dedup"] = {s: [len(dedup[s][0]), len(dedup[s][1])] for s in ABS_SETS}
    return O, E, uv, A


def perm_slotmap(d, rng):
    slot = d.slot.values.copy()
    for ix in d.groupby("podkey").indices.values():
        su = np.unique(slot[ix]); mp = dict(zip(su, rng.permutation(su)))
        slot[ix] = [mp[s] for s in d.slot.values[ix]]
    return d.pod.values, slot


def cooccupancy(d, pod, slot):
    """Share of drives that share their (pod, slot) with a drive whose residence overlaps theirs."""
    import pandas as pd
    x = pd.DataFrame({"p": d.podkey.values, "s": slot, "f": d["first"].values, "l": d["last"].values})
    bad = 0
    for _, g in x.groupby(["p", "s"]):
        if len(g) < 2:
            continue
        g = g.sort_values("f"); f, l = g.f.values, g.l.values
        bad += int(((f[1:] <= np.maximum.accumulate(l)[:-1])).sum())
    return bad / len(x)


def init(paths):
    global PER
    PER = [r2.prep(p) for p in paths]


def one(arg):
    kind, i = arg
    rng = np.random.default_rng([SEED, {"main": 0, "C1": 1, "C3": 3}[kind], i])
    tot = []; keep = []; co = None
    for d in PER:
        if kind == "main":
            pod, slot = r2.perm_main(d, rng)
        elif kind == "C1":
            pod, slot = r2.perm_slot(d, rng, "podkey")
            if i < 5:
                co = (co or []) + [cooccupancy(d, pod, slot)]
        else:
            pod, slot = perm_slotmap(d, rng)
        O, E, _, _ = tally(d, pod, slot)
        tot.append((O, E))
        keep.append(np.array([[O[SETS.index(s), WL.index(w)], E[SETS.index(s), WL.index(w)]] for s, w in PL_KEEP]))
    return kind, i, [t[0].sum(2) for t in tot], [t[1].sum(2) for t in tot], keep, co


def exch(d):
    out = {}
    sz = d.groupby("stratum").vault.transform("size")
    out["main_share_drives_in_strata_ge2"] = float((sz >= 2).mean())
    out["main_share_drives_in_strata_ge2_pods"] = float((d.groupby("stratum").pod.transform("nunique") >= 2).mean())
    sz2 = d.groupby("podstratum").pod.transform("size")
    out["C2_share_drives_in_pod_strata_ge2"] = float((sz2 >= 2).mean())
    out["median_main_stratum_size"] = float(sz.median())
    return out


def main(prefix, paths):
    init(paths)
    obs = [tally(d, d.pod.values, d.slot.values, detail=True) for d in PER]
    meta = {"periods": [], "sets": SETS, "wins": WL, "abs_sets": ABS_SETS, "pl_keep": PL_KEEP,
            "npl_main": NPL_MAIN, "npl_pod": NPL_POD, "seed": SEED}
    arr = {}
    for pi, (d, (O, E, uv, A)) in enumerate(zip(PER, obs)):
        g = d.groupby("vault").agg(dc=("dc", "first"), cl=("cluster", "first"), age=("poh0", "median"),
                                   n=("vault", "size"), npod=("podkey", "nunique")).reindex(uv)
        meta["periods"].append({"file": d.period.iloc[0], "n_index": int(d.fail.sum()), "n_drives": int(len(d)),
                                "dedup": A["dedup"], "exch": exch(d),
                                "vault": [float(x) for x in uv], "dc": [str(x) for x in g.dc.values],
                                "cluster": [int(x) for x in g.cl.values], "age_y": [float(x) / 8766 for x in g.age.values],
                                "npod": [int(x) for x in g.npod.values]})
        arr[f"O{pi}"], arr[f"E{pi}"] = O, E
        for kk in ("exp", "ev", "eexp"):
            arr[f"A{kk}{pi}"] = A[kk]
    tasks = [("main", i) for i in range(NPL_MAIN)] + [(k, i) for k in ("C1", "C3") for i in range(NPL_POD)]
    res = {k: {"tot_O": [], "tot_E": [], "keep": []} for k in ("main", "C1", "C3")}
    co = []
    with Pool(4, initializer=init, initargs=(paths,)) as pool:
        for n_done, (kind, i, tO, tE, keep, c) in enumerate(pool.imap_unordered(one, tasks, chunksize=4)):
            res[kind]["tot_O"].append(tO); res[kind]["tot_E"].append(tE); res[kind]["keep"].append(keep)
            if c:
                co += c
            if n_done % 50 == 0:
                print("done", n_done, "of", len(tasks), flush=True)
    for kind, r in res.items():
        for pi in range(len(PER)):
            arr[f"pl_{kind}_O{pi}"] = np.array([x[pi] for x in r["tot_O"]])
            arr[f"pl_{kind}_E{pi}"] = np.array([x[pi] for x in r["tot_E"]])
            arr[f"pl_{kind}_keep{pi}"] = np.array([x[pi] for x in r["keep"]])
    meta["C1_cooccupancy_share"] = co
    np.savez_compressed(prefix + ".npz", **arr)
    json.dump(meta, open(prefix + "_meta.json", "w"), indent=1)
    print("saved", prefix)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
