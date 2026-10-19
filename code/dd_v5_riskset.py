"""Index-time risk set and 2026H1 operational checks (fourth review, 2026-10-18). Written before running.

The main analysis (dd_unified.py) counts, for an index failure on day t, every other drive of the vault that
is observed at any time in the window, with its exposure clipped to its own observation span, and places it
by its last location in the period. So a drive that enters after t contributes from its first day, and a
drive that moved within the period is placed where it ended.
Variant RS (index-time risk set) keeps only member drives that
  - were present on the index day (first <= t <= last) and had not failed before t,
  - never changed (cluster, vault, pod) in the period (n_pods_seen == 1), so the location on day t is the
    location used,
and index failures whose own drive never moved. Exposure ends at the drive's last observation (exit without
failure = censored); failures count only inside the window.
Q1 / Q2: index failures on period days 0-89 / 90-180 (meaningful for 2026H1 only).
XANOM (2026H1 only): drop all drives of models whose 2026 Q1 failure rate exceeded 1.5x their 2025
rate (models with >= 500 drives), as a proxy for the unnamed model of the 2026 Q1 power-cycling issue
(Backblaze Drive Stats Q1 2026). The list is computed in this script and saved.
Placebos: main (pod, slot) within vault x model x age bin, NPL_MAIN draws; slot-map within pod, NPL_POD
draws. Output <out>.npz + <out>_meta.json, analysed by dd_v5_analyse.py.
USAGE  python dd_v5_riskset.py <out_prefix> <2025.parquet> <period.parquet> [...]   (last file = 2026H1)
"""
import sys, os, json, importlib.util
import numpy as np, pandas as pd
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("r2", os.path.join(HERE, "dd_review2.py"))
r2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(r2)
spec = importlib.util.spec_from_file_location("du", os.path.join(HERE, "dd_unified.py"))
du = importlib.util.module_from_spec(spec); spec.loader.exec_module(du)

NPL_MAIN = int(os.environ.get("NPL_MAIN", 300)); NPL_POD = int(os.environ.get("NPL_POD", 100))
SEED = 20261020
WINS = {"P": (1, 30), "L1": (1, 3), "L5": (31, 60), "G1": (61, 120), "G2": (121, 180)}
WL = list(WINS)
SETS = ["POD", "d0", "d1", "d2", "d3", "d4", "TOME", "REST"]
DB = [(0, 0), (1, 2), (3, 5), (6, 15), (16, 10 ** 6)]
VAR = ["ALL", "RS", "Q1", "Q2"]
DATA = None


def tally(d, pod, slot, detail=False):
    uv = np.unique(d.vault.values); nv = len(uv); vidx = {v: k for k, v in enumerate(uv)}
    O = np.zeros((len(VAR), len(SETS), len(WL), nv)); E = np.zeros_like(O)
    A = {k: np.zeros((len(VAR), len(SETS), nv)) for k in ("exp", "ev", "cens")}
    first, last, fd, h = d["first"].values, d["last"].values, d.fdate.values, d.h.values
    still = (d.n_pods_seen.values == 1)
    pend = d.pend.values[0]
    wa = np.array([WINS[w][0] for w in WL]); wb = np.array([WINS[w][1] for w in WL])
    for v, ix in d.groupby("vault").indices.items():
        k = vidx[v]
        f_, l_, fd_, h_, p_, s_, st_ = first[ix], last[ix], fd[ix], h[ix], pod[ix], slot[ix], still[ix]
        n = len(ix)
        for j in np.where(~np.isnan(fd_))[0]:
            t = fd_[j]
            other = np.ones(n, bool); other[j] = False
            sp = p_ == p_[j]; ss = s_ == s_[j]; dist = np.abs(s_ - s_[j])
            pm = other & sp
            M = [pm] + [pm & (dist >= lo) & (dist <= hi) for lo, hi in DB] + [other & ss & ~sp, other & ~sp & ~ss]
            M = np.array(M, dtype=float)
            present = (f_ <= t) & (l_ >= t) & ~(fd_ < t)
            rs = (present & st_).astype(float)
            st = np.maximum(f_[None, :], t + wa[:, None]); en = np.minimum(l_[None, :], t + wb[:, None])
            DD = np.clip(en - st + 1, 0, None)
            EV = ((fd_[None, :] >= t + wa[:, None]) & (fd_[None, :] <= t + wb[:, None])).astype(float)
            vmask = {"ALL": 1.0, "RS": rs if st_[j] else None, "Q1": 1.0 if t <= 89 else None, "Q2": 1.0 if t >= 90 else None}
            full = t <= pend - 30
            for vi, vn in enumerate(VAR):
                mm = vmask[vn]
                if mm is None:
                    continue
                Mv = M * mm
                O[vi, :, :, k] += Mv @ EV.T
                E[vi, :, :, k] += (Mv * h_[None, :]) @ DD.T
                if detail and full:
                    alive = (l_ >= t + 1) & ~(fd_ <= t)
                    ev30 = EV[0] > 0
                    cens = alive & ~ev30 & (l_ < t + 30)
                    for si in range(len(SETS)):
                        am = (Mv[si] > 0) & alive
                        A["exp"][vi, si, k] += am.sum(); A["ev"][vi, si, k] += (am & ev30).sum()
                        A["cens"][vi, si, k] += (am & cens).sum()
    return O, E, uv, A


def init(paths):
    global DATA
    DATA = load(paths)


def load(paths):
    p25, rest = paths[0], paths[1:]
    per = [r2.prep(p) for p in rest]
    flagged = anomalous_models(p25, rest[-1])
    raw = pd.read_parquet(rest[-1]); raw = raw[~raw.model.isin(flagged)]
    tmp = os.path.join(os.environ.get("V5_TMP", "/tmp"), os.path.basename(rest[-1]) + ".xanom.parquet")
    if not os.path.exists(tmp):
        raw.to_parquet(tmp)
    xa = r2.prep(tmp); xa["period"] = "k2026c_xanom"
    return {"periods": per, "xanom": xa, "flagged": flagged}


def anomalous_models(p25, p26):
    def rates(f, a, b):
        d = pd.read_parquet(f).dropna(subset=["slot"]); p0 = pd.Timestamp(d.period_start.iloc[0])
        fi = (pd.to_datetime(d["first"]) - p0).dt.days; la = (pd.to_datetime(d["last"]) - p0).dt.days
        fd = (pd.to_datetime(d.fdate) - p0).dt.days
        dd = (np.minimum(la, b) - np.maximum(fi, a) + 1).clip(lower=0)
        ff = (fd >= a) & (fd <= b) & (d.fail == 1)
        g = pd.DataFrame({"m": d.model, "dd": dd, "f": ff}).groupby("m").agg(dd=("dd", "sum"), f=("f", "sum"))
        return g.f / g.dd, g.dd / (b - a + 1)
    r25, _ = rates(p25, 0, 10 ** 4)
    out = set()
    for a, b in ((0, 89),):
        r, n = rates(p26, a, b)
        ok = (n >= 500) & (r > 1.5 * r25.reindex(r.index))
        out |= set(r.index[ok.fillna(False)])
    return sorted(out)


def one(arg):
    kind, i = arg
    rng = np.random.default_rng([SEED, {"main": 0, "C3": 3}[kind], i])
    res = []
    for d in DATA["periods"] + [DATA["xanom"]]:
        pod, slot = r2.perm_main(d, rng) if kind == "main" else du.perm_slotmap(d, rng)
        O, E, _, _ = tally(d, pod, slot)
        res.append((O.sum(-1), E.sum(-1), O[:, :, 0, :][:, [2, 5]], E[:, :, 0, :][:, [2, 5]]))
    return kind, res


def main(prefix, paths):
    init(paths)
    allp = DATA["periods"] + [DATA["xanom"]]
    arr = {}; meta = {"flagged_models": DATA["flagged"], "periods": [], "sets": SETS, "wins": WL, "var": VAR}
    for pi, d in enumerate(allp):
        O, E, uv, A = tally(d, d.pod.values, d.slot.values, detail=True)
        arr[f"O{pi}"], arr[f"E{pi}"] = O, E
        for k in A:
            arr[f"A{k}{pi}"] = A[k]
        meta["periods"].append({"file": str(d.period.iloc[0]), "vault": [float(x) for x in uv], "n_drives": int(len(d)),
                                "n_moved": int((d.n_pods_seen > 1).sum()), "n_fail": int(d.fail.sum()),
                                "n_fail_moved": int(((d.n_pods_seen > 1) & (d.fail == 1)).sum())})
    tasks = [("main", i) for i in range(NPL_MAIN)] + [("C3", i) for i in range(NPL_POD)]
    pl = {"main": [], "C3": []}
    with Pool(4, initializer=init, initargs=(paths,)) as pool:
        for n_done, (kind, r) in enumerate(pool.imap_unordered(one, tasks, chunksize=4)):
            pl[kind].append(r)
            if n_done % 50 == 0:
                print("done", n_done, "of", len(tasks), flush=True)
    for kind, lst in pl.items():
        for pi in range(len(allp)):
            arr[f"pl_{kind}_O{pi}"] = np.array([x[pi][0] for x in lst]); arr[f"pl_{kind}_E{pi}"] = np.array([x[pi][1] for x in lst])
            arr[f"pl_{kind}_vO{pi}"] = np.array([x[pi][2] for x in lst]); arr[f"pl_{kind}_vE{pi}"] = np.array([x[pi][3] for x in lst])
    np.savez_compressed(prefix + ".npz", **arr)
    json.dump(meta, open(prefix + "_meta.json", "w"), indent=1)
    print("saved", prefix, "flagged", DATA["flagged"])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
