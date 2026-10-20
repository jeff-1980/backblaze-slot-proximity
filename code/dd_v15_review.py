"""Revision-stage exploratory analyses answering the pre-submission mock review of manuscript v14 (2026-10-20).
Written before running; a two-draw test run (NPL=2) was inspected for errors before the 300-draw run, and no analysis
choice was changed after seeing estimates. Nothing here is pre-registered; every result is exploratory.

Base: clock F of dd_v10_clock.py (failure record F; a failing drive's exposure ends at F; daily baseline hazard per
model x age bin x vault recomputed on that exposure), window 1-30 days unless stated, sets POD, d0 (same slot, other
drives), d1 (slot distance 1-2, "near") and d4 (distance >= 16, "far"). Placebo: (pod, slot) permuted jointly within
vault x model x age bin (NPL draws, SEED 20261020). X = (O/E) / placebo mean of O/E; R = X_near / X_far.
Intervals: joint vault bootstrap (B = 1000) across periods (vault key = cluster * 100000 + vault_raw), placebo mean
held fixed. Groups: exploration (2023H2, 2024, 2025), hold-out (2026H1), pooled (all four).
A1 Leave-one-vault-out: R with each distinct vault (all its periods) removed, full-run placebo means held fixed.
A2 Subgroups (forest-plot groups: all, period, data centre, cluster 0 vs others; also cluster x capacity class of the
   index server): near/far O, E, subgroup placebo mean, X and R. The capacity class is that of dd_sample_flow.py
   (smallest of 26, 45, 60 at least the highest slot number seen + 1), grouped 60 / 45 / other; under the placebo
   the class is that of the server the index drive is placed in.
A3 Service-visit anchor. Rrep = first day another drive is recorded in the index drive's own (cluster, vault, pod,
   slot) after its last located day (dd_v10_clock.lr). Each 1-30-day window is split at Rrep into days before Rrep
   and days on/after Rrep (O by follow-on date, E by exposure days); index failures without Rrep in the window go
   wholly to "before" and are also reported alone. Server proxy: Srv = first day after the last located day on which
   a drive not seen earlier in the period in the index server (cluster, vault, pod) is first recorded there; same
   split, plus an index-level split (any Srv within the window or not). Placebo means: full-window base run.
A4 Early life. (a) Age at failure of near/far follow-on drives = poh0 / 24 + days from first record to F (power-on
   hours at first appearance plus calendar days). (b) Drives younger than 90 days (same age definition; unknown poh0
   kept) removed from follow-on counting and exposure, baseline hazard recomputed; index failures unchanged.
   (c) First age bin (0-182 days) split into 0-30, 31-90, 91-182 days in the hazard and placebo strata.
   Both (b) and (c) have their own placebo runs (same draws as base for (b); own strata for (c)).
A5 Absolute scale: excess per 1,000 index failures = 1000 (O - E * placebo mean) / n_index (index failures whose
   window starts inside the period), windows 1-30 and 1-3 days; distinct follow-on drives.
A6 Provenance items read from existing result files (sources named in the output) plus recomputed counts.
USAGE  NPL=300 python dd_v15_review.py <out.json> <runs_dir> <k2023h2c> <k2024> <k2025> <k2026c>
"""
import sys, os, json, re, importlib.util, time
import numpy as np, pandas as pd
import multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, fn):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, fn))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


r2 = _load("r2", "dd_review2.py")
v10 = _load("v10", "dd_v10_clock.py")

NPL = int(os.environ.get("NPL", 300)); B = 1000; SEED = 20261020
WA = np.array([1, 1]); WB = np.array([30, 3]); WN = ["w1_30", "w1_3"]
SETS = ["POD", "d0", "d1", "d4"]; S_POD, S_D0, S_D1, S_D4 = range(4)
VARS = ["base", "b90", "c_age"]
CLS = ["60", "45", "other"]
LAB = ["2023H2", "2024", "2025", "2026H1"]
GROUPS = {"explore": [0, 1, 2], "confirm": [3], "pooled": [0, 1, 2, 3]}
DATA = None
T0 = time.time()


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def server_event(runs_path, d, p0, L):
    """Srv: first day after L on which a drive not seen earlier in the period in the same (cluster, vault, pod) is
    first recorded there (the index drive itself excluded)."""
    r = pd.read_parquet(runs_path, columns=["s", "c", "v", "p", "t0"]); r["vault"] = r.c * 100000 + r.v
    r["t0"] = (pd.to_datetime(r.t0) - p0).dt.days
    arr = r.groupby(["vault", "p", "s"]).t0.min().reset_index()
    arr = arr[arr.t0 > 0]
    loc = r.merge(L.rename("L"), left_on="s", right_index=True)
    loc = loc[loc.t0 <= loc.L].sort_values("t0").groupby("s").tail(1)[["s", "vault", "p", "L"]]
    m = loc.merge(arr, on=["vault", "p"], suffixes=("", "_o"))
    m = m[(m.s_o != m.s) & (m.t0 > m.L)]
    return m.groupby("s").t0.min()


def hazard(model, agebin, vault, fail, days):
    x = pd.DataFrame({"m": model, "a": agebin, "v": vault, "f": fail, "dd": days})
    g = x.groupby(["m", "a", "v"]).agg(f=("f", "sum"), dd=("dd", "sum"))
    h = (g.f / g.dd.where(g.dd > 0)).fillna(0.0).rename("h")
    return x.join(h, on=["m", "a", "v"]).h.values


def strata_ix(d, col):
    return [ix for ix in d.groupby(col).indices.values() if len(ix) > 1]


def build(kpath, runs_dir):
    d = r2.prep(kpath)
    p0 = pd.Timestamp(pd.read_parquet(kpath, columns=["period_start"]).period_start.iloc[0])
    rp = os.path.join(runs_dir, v10.RUNS[os.path.basename(kpath)])
    t = v10.lr(rp, d, p0)
    d = d.join(t, on="serial_number")
    d["Srv"] = d.serial_number.map(server_event(rp, d, p0, t.L))
    F = d.fdate.values.astype(float)
    last = np.where(np.isnan(F), d["last"].values, np.minimum(d["last"].values, F)).astype(float)
    first = np.minimum(d["first"].values, last).astype(float)
    fail = (~np.isnan(F)).astype(int)
    P = {"n": len(d), "pend": int(d.pend.iloc[0]), "F": F, "pod": d.pod.values.astype(float), "slot": d.slot.values.astype(float),
         "vault": d.vault.values.astype(float), "Rrep": d.R.values.astype(float), "L": d.L.values.astype(float),
         "Srv": d.Srv.values.astype(float), "poh0": d.poh0.values.astype(float), "first0": d["first"].values.astype(float),
         "serial": d.serial_number.values, "label": os.path.basename(kpath)}
    uv, vid = np.unique(P["vault"], return_inverse=True); P["uv"], P["vid"] = uv, vid
    # base (clock F of dd_v10_clock.build)
    hb = hazard(d.model.values, d.agebin.values, d.vault.values, fail, last - first + 1)
    P["base"] = (first, last, F, hb)
    # (b) drop the first 90 days of service from exposure and follow-on counting
    a0 = P["poh0"] / 24.0
    s90 = np.where(np.isnan(a0), first, np.maximum(first, P["first0"] + np.ceil(90 - a0)))
    Fb = np.where(F >= s90, F, np.nan)
    hb90 = hazard(d.model.values, d.agebin.values, d.vault.values, (~np.isnan(Fb)).astype(int), np.clip(last - s90 + 1, 0, None))
    P["b90"] = (s90, last, Fb, hb90)
    P["b90_meta"] = {"drives_unknown_poh0_kept": int(np.isnan(a0).sum()), "drives_with_exposure_removed": int((s90 > first).sum()),
                     "drives_fully_removed": int((s90 > last).sum()), "failures_removed_from_follow_on": int((~np.isnan(F) & np.isnan(Fb)).sum()),
                     "failures_total": int((~np.isnan(F)).sum())}
    # (c) first age bin split 0-30 / 31-90 / 91-182 days
    ab = d.agebin.values.copy(); ad = d.poh0.fillna(0).values / 24.0
    abc = np.where(ab == 0, np.where(ad < 31, -3, np.where(ad < 91, -2, -1)), ab)
    hc = hazard(d.model.values, abc, d.vault.values, fail, last - first + 1)
    P["c_age"] = (first, last, F, hc)
    d["abc"] = abc
    P["strata_base"] = strata_ix(d, "stratum")
    P["strata_c"] = strata_ix(d, ["vault", "model", "abc"])
    P["c_meta"] = {"drives_bin0": int((ab == 0).sum()), "bin_0_30": int((abc == -3).sum()), "bin_31_90": int((abc == -2).sum()),
                   "bin_91_182": int((abc == -1).sum())}
    # capacity class of each server (dd_sample_flow.py), 0 = 60, 1 = 45, 2 = other
    top = d.groupby(["vault", "pod"]).slot.max() + 1
    cl = pd.cut(top, [0, 26, 45, 60, 1000], labels=["26", "45", "60", ">60"]).astype(str)
    code = cl.map({"60": 0, "45": 1}).fillna(2).astype(int)
    P["cls_map"] = {(float(v), float(p)): int(c) for (v, p), c in code.items()}
    P["server_classes"] = cl.value_counts().to_dict()
    vd = d.groupby("vault").agg(dc=("dc", "first"), cl=("cluster", "first"), vr=("vault_raw", "first")).reindex(uv)
    P["vdc"], P["vcl"], P["vraw"] = vd.dc.values.astype(str), vd.cl.values.astype(int), vd.vr.values.astype(int)
    return P


def perm(P, strata, rng):
    pod = P["pod"].copy(); slot = P["slot"].copy()
    for ix in strata:
        p = rng.permutation(ix); pod[ix], slot[ix] = P["pod"][p], P["slot"][p]
    return pod, slot


def tally(P, pod, slot, var, obs=False):
    """Per index failure: OE[n, O/E, set, window]; with obs also the window splits and the counted pairs."""
    mf, ml, mfd, h = P[var]
    ifd = P["F"]; vid = P["vid"]
    key = vid.astype(np.int64) * 1000 + pod.astype(np.int64)
    order = np.argsort(key, kind="stable"); uk, start, cnt = np.unique(key[order], return_index=True, return_counts=True)
    gpos = np.searchsorted(uk, key)
    idx = np.where(~np.isnan(ifd))[0]; ni = len(idx)
    OE = np.zeros((ni, 2, 4, 2))
    if obs:
        SP = np.zeros((ni, 2, 2, 2, 4))     # split (Rrep, Srv), part (before, on/after), O/E, set
        pairs = []
    for n_, j in enumerate(idx):
        g = gpos[j]; mem = order[start[g]:start[g] + cnt[g]]; mem = mem[mem != j]
        t = ifd[j]
        dist = np.abs(slot[mem] - slot[j])
        M = np.stack([np.ones(len(mem), bool), dist == 0, (dist >= 1) & (dist <= 2), dist >= 16]).astype(float)
        f_, l_, fd_, h_ = mf[mem], ml[mem], mfd[mem], h[mem]
        st = np.maximum(f_[None, :], t + WA[:, None]); en = np.minimum(l_[None, :], t + WB[:, None])
        DD = np.clip(en - st + 1, 0, None)
        EV = ((fd_[None, :] >= t + WA[:, None]) & (fd_[None, :] <= t + WB[:, None])).astype(float)
        OE[n_, 0] = M @ EV.T
        Mh = M * h_[None, :]
        OE[n_, 1] = Mh @ DD.T
        if obs:
            for k, cut in enumerate((P["Rrep"][j], P["Srv"][j])):
                c = np.inf if np.isnan(cut) else cut
                for part, (lo, hi) in enumerate(((t + 1, min(t + 30, c - 1)), (max(t + 1, c), t + 30))):
                    if hi < lo:
                        continue
                    dd = np.clip(np.minimum(l_, hi) - np.maximum(f_, lo) + 1, 0, None)
                    ev = ((fd_ >= lo) & (fd_ <= hi)).astype(float)
                    SP[n_, k, part, 0] = M @ ev; SP[n_, k, part, 1] = Mh @ dd
            for si in (S_D0, S_D1, S_D4):
                for w in (0, 1):
                    for q in np.where((M[si] > 0) & (EV[w] > 0))[0]:
                        pairs.append((n_, int(mem[q]), si, w))
    cm = P["cls_map"]
    cls = np.array([cm[(P["vault"][j], pod[j])] for j in idx], int)
    out = {"OE": OE, "idx": idx, "vix": vid[idx], "cls": cls, "ni_ok": (ifd[idx] + 1 <= P["pend"])}
    if obs:
        out["SP"] = SP; out["pairs"] = np.array(pairs, dtype=np.int64).reshape(-1, 4)
    return out


def agg_vc(P, T):
    """Per (class, set, window, O/E, vault) sums, float32."""
    nv = len(P["uv"]); A = np.zeros((3, 4, 2, 2, nv), np.float32)
    for c in range(3):
        m = T["cls"] == c
        if not m.any():
            continue
        for s in range(4):
            for w in range(2):
                for k in range(2):
                    A[c, s, w, k] = np.bincount(T["vix"][m], weights=T["OE"][m, k, s, w], minlength=nv)
    return A


def one(i):
    rng = np.random.default_rng([SEED, i]); out = []
    for P in DATA:
        pod, slot = perm(P, P["strata_base"], rng)
        tb = tally(P, pod, slot, "base"); t9 = tally(P, pod, slot, "b90")
        pod_c, slot_c = perm(P, P["strata_c"], rng)
        tc = tally(P, pod_c, slot_c, "c_age")
        out.append((agg_vc(P, tb), t9["OE"].sum(0).astype(np.float32), tc["OE"].sum(0).astype(np.float32)))
    return i, out


# ----------------------------------------------------------------------------------------------- analysis helpers
def pct(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))) if len(x) else (None, None)


class Ctx:
    def __init__(self, D, OBS, PL):
        self.D, self.OBS, self.PL = D, OBS, PL
        self.U = np.unique(np.concatenate([P["uv"] for P in D]))
        self.uidx = [np.searchsorted(self.U, P["uv"]) for P in D]
        rng = np.random.default_rng(SEED)
        self.mult = np.stack([np.bincount(rng.integers(0, len(self.U), len(self.U)), minlength=len(self.U)) for _ in range(B)]).astype(float)

    def boot(self, ps, vecs):
        return sum(self.mult[:, self.uidx[p]] @ vecs[p] for p in ps)

    def pm(self, ps, s, w, vmask=None, cmask=(0, 1, 2), var="base"):
        if var == "base":
            po = sum(self.PL[p][0][:, list(cmask), s, w, 0, :][..., (vmask[p] if vmask else slice(None))].sum((1, 2)) for p in ps)
            pe = sum(self.PL[p][0][:, list(cmask), s, w, 1, :][..., (vmask[p] if vmask else slice(None))].sum((1, 2)) for p in ps)
        else:
            k = 1 if var == "b90" else 2
            po = sum(self.PL[p][k][:, 0, s, w] for p in ps); pe = sum(self.PL[p][k][:, 1, s, w] for p in ps)
        r = po.astype(float) / np.maximum(pe.astype(float), 1e-12)
        return float(r.mean()), float(r.std() / np.sqrt(len(r)))


def main(outp, runs_dir, paths):
    global DATA
    log("building", NPL, "placebo draws planned")
    D = [build(p, runs_dir) for p in paths]
    for P in D:
        log("built", P["label"], P["n"], "drives", int((~np.isnan(P["F"])).sum()), "failures")
    OBS = []
    for P in D:
        o = {"base": tally(P, P["pod"], P["slot"], "base", obs=True)}
        for v in ("b90", "c_age"):
            o[v] = tally(P, P["pod"], P["slot"], v)
        OBS.append(o)
        log("observed tally", P["label"])
    DATA = D
    PLr = [[None] * NPL for _ in D]
    with mp.get_context("fork").Pool(4) as pool:
        for k, (i, out) in enumerate(pool.imap_unordered(one, range(NPL), chunksize=1)):
            for p in range(len(D)):
                PLr[p][i] = out[p]
            if k % 10 == 0 or k == NPL - 1:
                log("placebo", k + 1, "of", NPL)
    PL = [(np.stack([x[0] for x in PLr[p]]), np.stack([x[1] for x in PLr[p]]), np.stack([x[2] for x in PLr[p]])) for p in range(len(D))]
    C = Ctx(D, OBS, PL)
    res = {"meta": {"script": "code/dd_v15_review.py", "NPL": NPL, "B": B, "SEED": SEED, "clock": "F (dd_v10_clock.py)",
                    "status": "revision-stage exploratory analysis, written before running",
                    "periods": [P["label"] for P in D],
                    "n_index_failures": {P["label"]: int(len(OBS[p]["base"]["idx"])) for p, P in enumerate(D)},
                    "n_index_window_in_period": {P["label"]: int(OBS[p]["base"]["ni_ok"].sum()) for p, P in enumerate(D)}}}

    # ---------- core estimator: X and R for a selection of index failures
    def est(ps, var="base", w=0, isel=None, vmask=None, cmask=(0, 1, 2), oe=None, pm_full=False, sets=(S_D1, S_D4)):
        """isel: per-period index-failure mask; oe: per-period array (ni, 2, 4) replacing OE[..., w]."""
        out = {}; xb = {}
        for s in sets:
            vo, ve = {}, {}
            for p in ps:
                T = OBS[p][var]; A = oe[p] if oe is not None else T["OE"][:, :, :, w]
                m = np.ones(len(T["idx"]), bool) if isel is None else isel[p]
                if vmask is not None:
                    m = m & vmask[p][T["vix"]]
                m = m & np.isin(T["cls"], cmask)
                vo[p] = np.bincount(T["vix"][m], weights=A[m, 0, s], minlength=len(D[p]["uv"]))
                ve[p] = np.bincount(T["vix"][m], weights=A[m, 1, s], minlength=len(D[p]["uv"]))
            O = sum(vo[p].sum() for p in ps); E = sum(ve[p].sum() for p in ps)
            if pm_full or vmask is None and cmask == (0, 1, 2):
                pm, se = C.pm(ps, s, w, var=var)
            else:
                pm, se = C.pm(ps, s, w, vmask=vmask, cmask=cmask, var=var)
            bo, be = C.boot(ps, vo), C.boot(ps, ve)
            x = O / E / pm if E > 0 else float("nan")
            xb[s] = bo / np.maximum(be, 1e-12) / pm
            lo, hi = pct(xb[s])
            out[SETS[s]] = {"O": int(round(O)), "E": float(E), "placebo_mean": pm, "placebo_mc_se": se, "X": float(x), "lo": lo, "hi": hi}
            out[SETS[s]]["_vo"], out[SETS[s]]["_ve"], out[SETS[s]]["_pm"] = vo, ve, pm
        if S_D1 in sets and S_D4 in sets:
            r = out["d1"]["X"] / out["d4"]["X"] if out["d4"]["X"] > 0 else float("nan")
            lo, hi = pct(xb[S_D1] / np.maximum(xb[S_D4], 1e-12))
            small = min(out["d1"]["O"], out["d4"]["O"]) < 10
            out["R"] = {"ratio": float(r), "lo": lo, "hi": hi, "ci_unreliable_fewer_than_10_near_or_far": bool(small)}
        return out

    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items() if not str(k).startswith("_")}
        return o

    # ---------- reproduction check
    rep = {}
    U = json.load(open(os.path.join(HERE, "..", "results", "unified_rekeyc.json")))
    CK = json.load(open(os.path.join(HERE, "..", "results", "v10", "clock.json")))
    base = {}
    for g, ps in GROUPS.items():
        base[g] = est(ps, sets=(S_POD, S_D0, S_D1, S_D4))
        rep[g] = {"this_run": clean(base[g]),
                  "unified_rekeyc_H4": U["groups"][g]["hyp"]["H4_near_far"],
                  "v10_clock_F_ALL_w1_30": CK["groups"].get(g, {}).get("F_ALL", {}).get("w1_30")}
        log("reproduction", g, "R", round(base[g]["R"]["ratio"], 3), base[g]["R"]["lo"], base[g]["R"]["hi"],
            "near", base[g]["d1"]["O"], "far", base[g]["d4"]["O"])
    res["reproduction_check"] = rep

    # ---------- A1 leave-one-vault-out
    a1 = {"note": "Each distinct vault key (cluster*100000+vault_raw) removed from all periods of the group; placebo means of the full run held fixed (not re-estimated)."}
    for g, ps in (("explore", [0, 1, 2]), ("confirm", [3])):
        b = base[g]; pm1, pm4 = b["d1"]["_pm"], b["d4"]["_pm"]
        tot = {}
        for s in ("d1", "d4"):
            for k in ("_vo", "_ve"):
                v = np.zeros(len(C.U))
                for p in ps:
                    np.add.at(v, C.uidx[p], b[s][k][p])
                tot[s + k] = v
        O1, E1, O4, E4 = tot["d1_vo"], tot["d1_ve"], tot["d4_vo"], tot["d4_ve"]
        R0 = b["R"]["ratio"]
        present = np.zeros(len(C.U), bool)
        nvp = 0
        for p in ps:
            present[C.uidx[p]] = True; nvp += len(D[p]["uv"])
        ui = np.where(present)[0]
        Rm = ((O1.sum() - O1[ui]) / (E1.sum() - E1[ui]) / pm1) / ((O4.sum() - O4[ui]) / (E4.sum() - E4[ui]) / pm4)
        infl = np.argsort(-np.abs(Rm - R0))[:5]
        info = {}
        for p in ps:
            for k, u in enumerate(D[p]["uv"]):
                info.setdefault(float(u), (D[p]["vcl"][k], D[p]["vraw"][k], D[p]["vdc"][k]))
        top = []
        for k in infl:
            u = float(C.U[ui[k]]); cl, vr, dc = info[u]
            top.append({"vault_key": int(u), "cluster": int(cl), "vault": int(vr), "datacenter": dc, "near_events": int(O1[ui[k]]),
                        "far_events": int(O4[ui[k]]), "R_without": float(Rm[k]), "change": float(Rm[k] - R0)})
        # distinct vaults
        nv_index = len(np.unique(np.concatenate([C.U[C.uidx[p]][np.unique(OBS[p]["base"]["vix"])] for p in ps])))
        a1[g] = {"R_full": R0, "min": float(np.nanmin(Rm)), "max": float(np.nanmax(Rm)),
                 "vault_min": int(C.U[ui[np.nanargmin(Rm)]]), "vault_max": int(C.U[ui[np.nanargmax(Rm)]]),
                 "max_abs_change": float(np.nanmax(np.abs(Rm - R0))), "top5_influential": top,
                 "n_distinct_vaults": int(present.sum()), "n_vault_periods": int(nvp),
                 "n_distinct_vaults_with_index_failure": int(nv_index),
                 "n_distinct_vaults_with_near_event": int((O1[ui] > 0).sum()),
                 "sum_over_periods_of_vaults_with_near_event": int(sum((b["d1"]["_vo"][p] > 0).sum() for p in ps)),
                 "per_period_vaults_with_near_event": {LAB[p]: int((b["d1"]["_vo"][p] > 0).sum()) for p in ps}}
        log("A1", g, a1[g]["min"], a1[g]["max"], a1[g]["n_distinct_vaults_with_near_event"])
    res["A1_leave_one_vault_out"] = a1

    # ---------- A2 subgroups
    a2 = {"note": "R intervals of cells with fewer than 10 near or far events are flagged (ci_unreliable_...): resamples with no far event make the upper bound explode. Subgroup placebo means computed on the subgroup's vaults (and, for capacity class, on index failures placed in servers of that class), as in dd_unified_analyse.sub_est; vault bootstrap with placebo mean fixed. Rows with zero far or near events give undefined R."}
    for g, ps in GROUPS.items():
        G = {}
        def put(name, vmask=None, cmask=(0, 1, 2)):
            e = est(ps, vmask=vmask, cmask=cmask)
            nv = sum(int(vmask[p].sum()) if vmask is not None else len(D[p]["uv"]) for p in ps)
            G[name] = {**clean(e), "n_vault_periods": nv}
        put("all")
        if len(ps) > 1:
            for p in ps:
                put(f"period_{LAB[p]}", {q: (np.ones(len(D[q]["uv"]), bool) if q == p else np.zeros(len(D[q]["uv"]), bool)) for q in ps})
        dcs = sorted(set(np.concatenate([D[p]["vdc"] for p in ps])))
        for dc in dcs:
            put(f"dc_{dc}", {p: D[p]["vdc"] == dc for p in ps})
        put("cluster0", {p: D[p]["vcl"] == 0 for p in ps}); put("other_clusters", {p: D[p]["vcl"] != 0 for p in ps})
        for cname, cm in (("c0", lambda p: D[p]["vcl"] == 0), ("other", lambda p: D[p]["vcl"] != 0)):
            for k, cl in enumerate(CLS):
                put(f"{cname}_class{cl}", {p: cm(p) for p in ps}, (k,))
        for k, cl in enumerate(CLS):
            put(f"class{cl}", None, (k,))
        a2[g] = G
        log("A2", g, {k: (round(v["R"]["ratio"], 2) if v["R"]["ratio"] == v["R"]["ratio"] else None) for k, v in G.items()})
    res["A2_subgroups"] = a2

    # ---------- A3 service-visit anchor
    a3 = {"note": "Placebo means from the full-window base run (sets d1, d4, window 1-30) applied to every part. Rrep from dd_v10_clock.lr; Srv = first new drive in the index server after the index drive's last located day."}
    for g, ps in (("explore", [0, 1, 2]), ("confirm", [3])):
        A = {}
        for k, nm in ((0, "slot_replacement_Rrep"), (1, "server_new_drive_Srv")):
            cut = {p: (D[p]["Rrep"] if k == 0 else D[p]["Srv"])[OBS[p]["base"]["idx"]] for p in ps}
            t = {p: D[p]["F"][OBS[p]["base"]["idx"]] for p in ps}
            cat = {p: np.where(np.isnan(cut[p]) | (cut[p] > t[p] + 30), 0, np.where(cut[p] <= t[p] + 1, 2, 1)) for p in ps}
            sub = {"index_failures": {"no_event_in_window": int(sum((cat[p] == 0).sum() for p in ps)),
                                      "event_inside_window_day2_30": int(sum((cat[p] == 1).sum() for p in ps)),
                                      "event_on_or_before_day1": int(sum((cat[p] == 2).sum() for p in ps)),
                                      "event_missing": int(sum(np.isnan(cut[p]).sum() for p in ps))}}
            for part, pn in ((0, "before"), (1, "on_or_after")):
                oe = {p: OBS[p]["base"]["SP"][:, k, part] for p in ps}
                sub[pn] = clean(est(ps, oe=oe, pm_full=True))
            oe = {p: OBS[p]["base"]["SP"][:, k, 0] for p in ps}
            sub["no_event_in_window_only"] = clean(est(ps, oe=oe, isel={p: cat[p] == 0 for p in ps}, pm_full=True))
            sub["event_in_window_before_part"] = clean(est(ps, oe=oe, isel={p: cat[p] > 0 for p in ps}, pm_full=True))
            if k == 1:
                sub["index_level_with_server_event_in_window"] = clean(est(ps, isel={p: cat[p] > 0 for p in ps}, pm_full=True))
                sub["index_level_without_server_event_in_window"] = clean(est(ps, isel={p: cat[p] == 0 for p in ps}, pm_full=True))
            # follow-on date minus cut, near vs far
            dist = {}
            for si, sn in ((S_D1, "near"), (S_D4, "far")):
                diffs = []; nnone = 0
                for p in ps:
                    pr = OBS[p]["base"]["pairs"]; pr = pr[(pr[:, 2] == si) & (pr[:, 3] == 0)]
                    c = cut[p][pr[:, 0]]; fdq = D[p]["F"][pr[:, 1]]
                    nnone += int(np.isnan(c).sum()); diffs.append((fdq - c)[~np.isnan(c)])
                x = np.concatenate(diffs)
                dist[sn] = {"n_events": int(len(x) + nnone), "n_without_event_day": nnone,
                            "median": float(np.median(x)) if len(x) else None,
                            "q25": float(np.percentile(x, 25)) if len(x) else None, "q75": float(np.percentile(x, 75)) if len(x) else None,
                            "share_0_3": float(((x >= 0) & (x <= 3)).mean()) if len(x) else None,
                            "share_negative": float((x < 0).mean()) if len(x) else None}
            sub["followon_minus_event_day"] = dist
            A[nm] = sub
        a3[g] = A
        log("A3", g, {nm: (round(A[nm]["before"]["R"]["ratio"], 2), round(A[nm]["on_or_after"]["R"]["ratio"], 2)) for nm in A})
    res["A3_service_visit"] = a3

    # ---------- A4 early life
    a4 = {"age_definition": "age at failure = poh0/24 (power-on hours at first appearance in the period) + calendar days from first record to failure record F; days since first record in the period reported as well (left-censored at period start).",
          "b_meta": {D[p]["label"]: D[p]["b90_meta"] for p in range(len(D))},
          "c_meta": {D[p]["label"]: D[p]["c_meta"] for p in range(len(D))}}
    for g, ps in (("explore", [0, 1, 2]), ("confirm", [3])):
        A = {"a_age_at_failure": {}}
        for si, sn in ((S_D1, "near"), (S_D4, "far"), (S_D0, "d0")):
            ag, dsf, miss = [], [], 0
            for p in ps:
                pr = OBS[p]["base"]["pairs"]; pr = pr[(pr[:, 2] == si) & (pr[:, 3] == 0)]
                q = pr[:, 1]; a = D[p]["poh0"][q] / 24 + (D[p]["F"][q] - D[p]["first0"][q])
                miss += int(np.isnan(a).sum()); ag.append(a[~np.isnan(a)]); dsf.append(D[p]["F"][q] - D[p]["first0"][q])
            a = np.concatenate(ag); s = np.concatenate(dsf)
            A["a_age_at_failure"][sn] = {"n_events": int(len(a) + miss), "n_poh_missing": miss, "median_days": float(np.median(a)),
                                         "q25": float(np.percentile(a, 25)), "q75": float(np.percentile(a, 75)),
                                         "share_lt_90d": float((a < 90).mean()), "share_lt_182d": float((a < 182).mean()),
                                         "days_since_first_record_median": float(np.median(s)), "days_since_first_record_share_lt_90": float((s < 90).mean())}
        A["base"] = clean(est(ps, sets=(S_POD, S_D0, S_D1, S_D4)))
        A["b_exclude_first_90_days"] = clean(est(ps, var="b90", sets=(S_POD, S_D0, S_D1, S_D4)))
        A["c_split_first_age_bin"] = clean(est(ps, var="c_age", sets=(S_POD, S_D0, S_D1, S_D4)))
        a4[g] = A
        log("A4", g, "R b", round(A["b_exclude_first_90_days"]["R"]["ratio"], 3), "R c", round(A["c_split_first_age_bin"]["R"]["ratio"], 3))
    res["A4_early_life"] = a4

    # ---------- A5 absolute scale
    a5 = {"note": "Excess per 1,000 index failures = 1000*(O - E*placebo_mean)/n_index; n_index = index failures whose window starts inside the period. A follow-on failure is counted once for every index window that contains it (and for every index in its server), so these are event-clustering counts, not distinct drives."}
    for g, ps in (("explore", [0, 1, 2]), ("confirm", [3])):
        A = {}
        for w, wn in enumerate(WN):
            e = est(ps, w=w)
            nvec = {p: np.bincount(OBS[p]["base"]["vix"], weights=OBS[p]["base"]["ni_ok"].astype(float), minlength=len(D[p]["uv"])) for p in ps}
            nI = sum(nvec[p].sum() for p in ps); nb = C.boot(ps, nvec)
            W_ = {"n_index": int(nI)}
            for sn in ("d1", "d4"):
                x = e[sn]; pm = x["_pm"]
                ex = 1000 * (x["O"] - x["E"] * pm) / nI
                bex = 1000 * (C.boot(ps, x["_vo"]) - C.boot(ps, x["_ve"]) * pm) / nb
                lo, hi = pct(bex)
                W_[{"d1": "near", "d4": "far"}[sn]] = {"O": x["O"], "E": x["E"], "placebo_mean": pm, "O_per_1000": 1000 * x["O"] / nI,
                                                       "E_per_1000": 1000 * x["E"] / nI, "E_times_pm_per_1000": 1000 * x["E"] * pm / nI,
                                                       "excess_per_1000": ex, "lo": lo, "hi": hi}
            # distinct follow-on drives
            for si, sn in ((S_D1, "near"), (S_D4, "far"), (S_D0, "d0")):
                rows = []
                for p in ps:
                    pr = OBS[p]["base"]["pairs"]; pr = pr[(pr[:, 2] == si) & (pr[:, 3] == w)]
                    rows += [(p, int(q)) for q in pr[:, 1]]
                cnt = pd.Series(rows).value_counts() if rows else pd.Series(dtype=int)
                W_[f"{sn}_counted_events"] = len(rows); W_[f"{sn}_distinct_followon_drives"] = int(len(cnt))
                W_[f"{sn}_drives_counted_more_than_once"] = int((cnt > 1).sum())
            A[wn] = W_
        a5[g] = A
        log("A5", g, {wn: (round(A[wn]["near"]["excess_per_1000"], 2), A[wn]["near_distinct_followon_drives"]) for wn in A})
    res["A5_absolute"] = a5

    # ---------- A6 provenance
    res["A6_provenance"] = provenance(OBS, D, runs_dir, paths, a5)
    json.dump(res, open(outp, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    log("written", outp)


def provenance(OBS, D, runs_dir, paths, a5):
    R = os.path.join(HERE, "..", "results"); J = lambda f: json.load(open(os.path.join(R, f)))
    out = {}
    r2e, r2c = J("review2_explore.json"), J("review2_2026.json")
    out["i_distinct_near_followon_drives"] = {
        "recomputed_this_run": {"explore": a5["explore"]["w1_30"]["near_distinct_followon_drives"], "confirm": a5["confirm"]["w1_30"]["near_distinct_followon_drives"],
                                "explore_counted_events": a5["explore"]["w1_30"]["near_counted_events"], "confirm_counted_events": a5["confirm"]["w1_30"]["near_counted_events"]},
        "archived_source": {"file": "results/review2_explore.json and results/review2_2026.json, A_raw.d1.unique_followers (dd_review2.py)",
                            "explore": {k: r2e["A_raw"]["d1"][k] for k in ("O", "unique_followers", "followers_multi_index", "n_vaults_with_event")},
                            "confirm": {k: r2c["A_raw"]["d1"][k] for k in ("O", "unique_followers", "followers_multi_index", "n_vaults_with_event")}},
        "same_batch_counts": {"file": "results/unified_rekeyc.json groups.confirm.hyp.H7_samebatch.Oa (same-batch near events)",
                              "confirm_same_batch_near": J("unified_rekeyc.json")["groups"]["confirm"]["hyp"]["H7_samebatch"]["Oa"],
                              "confirm_near_total": J("unified_rekeyc.json")["groups"]["confirm"]["hyp"]["H4_near_far"]["Oa"]}}
    b26 = J("dd_batch_rekeyc_2026.json")["table"]
    out["i_distinct_near_followon_drives"]["same_batch_counts"]["dd_batch_rekeyc_2026.table"] = {k: b26.get(k, {}).get("O") for k in ("POD_d1", "POD_d1_SB", "POD_d1_OT")}
    # (ii)
    U = J("unified_rekeyc.json"); lg = J("v7/lags.json"); cm = J("v7/cohort_main.json")
    out["ii_index_totals"] = {
        "10061": {"review2_explore.n_index.FULL": r2e["n_index"]["FULL"], "unified_rekeyc.n_index_full": {x["file"]: x["n_index_full"] for x in U["n_index_full"]},
                  "rule": "failure date <= period end - 30 (complete 30-day window), all failures of drives with a slot number (dd_review2 FULL filter; dd_unified_analyse n_index_full)"},
        "10060": {"v7/lags.json n_failures": {k: v["n_failures"] for k, v in lg.items()},
                  "n_no_located_record_before_failure": {k: v["n_no_located_record_before_failure"] for k, v in lg.items()},
                  "rule": "same complete-window failures but only those with a located daily record on or before the failure date (dd_v7_lags.py); 2024 has one failing drive with no located record before its failure"},
        "hold_out_2145": {"review2_2026.n_index.FULL": r2c["n_index"]["FULL"], "v7/lags 2026": lg.get("k2026c.parquet", {}).get("n_failures")}}
    try:
        out["ii_index_totals"]["705_706"] = v7_d0_check(runs_dir, paths[3], cm)
    except Exception as ex:
        out["ii_index_totals"]["705_706"] = {"error": str(ex)}
    # (iii)
    idc = J("dd_idcheck_rekeyc_2.json")
    pc = [json.loads(l) for l in open(os.path.join(R, "dd_review2_pc_master.txt")) if l.startswith("{") and '"2023-12-31"' in l]
    rows = {"2023H2": pc[0]["rows"] if pc else None}; loc = {"2023H2": pc[0]["loc_days"] if pc else None}
    for x in idc:
        lb = {"2024": "2024", "2025": "2025", "2026": "2026H1"}[x["start"][:4]]; rows[lb] = x["rows"]; loc[lb] = x["loc_days"]
    rr = {}
    for f, lb in (("runs_k2023h2.json", "2023H2"), ("runs_k2024.json", "2024"), ("runs_k2025.json", "2025"), ("runs_k2026.json", "2026H1")):
        rr[lb] = json.load(open(os.path.join(runs_dir, f)))["rows"]
    out["iii_drive_days"] = {"daily_rows_with_server_location": rows, "sum_rows": sum(rows.values()) if None not in rows.values() else None,
                             "slot_location_days (distinct day x (cluster,vault,pod,slot))": loc, "sum_loc_days": sum(loc.values()) if None not in loc.values() else None,
                             "drive_days_with_slot (runs files rows)": rr, "sum_drive_days_with_slot": sum(rr.values()),
                             "sources": "rows/loc_days: results/dd_idcheck_rekeyc_2.json (2024-2026H1) and results/dd_review2_pc_master.txt line 2 (2023H2 full-period idcheck output); drive-days with slot: results/v6runs/runs_k*.json rows",
                             "reading": "333.7 M = all daily rows with a server (vault, pod) location; 329.0 M = drive-days with a slot number (location-days 328,978,216; drive-days 328,978,223; the difference of 7 is the seven 2025 location-days with two drives)."}
    # (iv)
    bf = J("dd_benefit_rekeyc.json")
    out["iv_drive_snapshot_metric"] = {"file": "results/dd_benefit_rekeyc.json (pre-specified drive-snapshot test, dd_benefit.py)",
                                       "auc": bf["auc"], "recall_capture": bf["recall"], "test_positives": bf["test_positives"], "n_test_rows": bf["n_test_rows"],
                                       "reading": bf.get("reading"), "note": "capture = share of positive test rows (drive-days followed by a failure) inside the top-ranked budget share of rows"}
    # (v)
    cd = J("dd_benefit_cooldown.json")
    out["v_random_tie_order"] = {"file": "results/dd_benefit_cooldown.json budgets.<b>.tie_random (dd_benefit_cooldown.py; 20 random tie orders)",
                                 "budgets": {b: {"row_order": {k: v["caught"] for k, v in cd["budgets"][b]["rows"].items()}, "tie_random": cd["budgets"][b]["tie_random"]}
                                             for b in cd["budgets"]},
                                 "key_names": "s0 = baseline score M0, s1 = neighbour model M1; c<k> = cool-down of k weekly scans (c0 = original rule); row_order = ties broken by row order (Supplementary Table 8); tie_random = 20 random tie orders (c0 and the primary M0 cool-down c1 only)"}
    # (vi)
    out["vi_original_key_columns"] = {
        "finding": "No result file in results/ or rerun_check/ holds the Supplementary Table 2 original-key values (explore H1 1.21 (1.14-1.29), H4 1.89 (1.59-2.24); 2026H1 H1 1.28 (1.13-1.45), H4 1.90 (1.37-2.58)). They come from the original frozen run of 2026-10-14, whose output files (dd_confirm_2026.json, dd_batch_diag_2026.json) were not archived. The only archived original-key frozen-script output is results/rerun_check/dd_confirm_origkey_2026-10-19.json (2026H1 only, current tables).",
        "rerun_check_2026_X": {k: J("rerun_check/dd_confirm_origkey_2026-10-19.json")["main"]["table"][k]["X"] for k in ("POD|P", "POD_d1|P", "POD_d4|P", "TOME|P")}}
    return out


def v7_d0_check(runs_dir, k2026, cm):
    os.environ.setdefault("REPL", "1")
    v7 = _load("v7", "dd_v7_cohort.py")
    sg, li, meta = v7.load(os.path.join(runs_dir, "runs_k2026.parquet"), k2026)
    sgv = sg.groupby("vault").indices
    n = {0: [0, 0], 2: [0, 0], 3: [0, 0]}; zero_blocked = []
    for j in range(len(li)):
        v = li.vault.values[j]; ix = sgv.get(v)
        t, sj, pj, slj, ty = li.t.values[j], li.s.values[j], li.p.values[j], li.sl.values[j], li.ityp.values[j]
        if ix is None:
            if ty == 3:
                zero_blocked.append({"serial": str(sj), "reason": "vault has no runs"})
            continue
        a0, a1, f = sg.t0.values[ix], sg.t1.values[ix], sg.f.values[ix]
        mem = (a0 <= t) & (a1 >= t) & (sg.s.values[ix] != sj) & ~(f == t)
        d0 = mem & (sg.p.values[ix] == pj) & (sg.sl.values[ix] == slj)
        n[ty][0] += 1; n[ty][1] += int(d0.sum())
        if ty == 3 and d0.sum() == 0:
            cand = (sg.p.values[ix] == pj) & (sg.sl.values[ix] == slj) & (sg.s.values[ix] != sj)
            zero_blocked.append({"serial": str(sj), "t": int(t), "other_runs_in_slot": [(str(s), int(x0), int(x1), None if np.isnan(ff) else int(ff))
                                 for s, x0, x1, ff in zip(sg.s.values[ix][cand], a0[cand], a1[cand], f[cand])][:5]})
    return {"archived": {"cohort_main.confirm.index_by_type": cm["groups"]["confirm"]["index_by_type"], "cohort_main.confirm.sets.d0.exposures": cm["groups"]["confirm"]["sets"]["d0"]["exposures"]},
            "recomputed_index_and_d0_exposures_by_index_type": {"obs": n[0], "ext": n[2], "ext_blocked": n[3]},
            "blocked_indices_without_d0_exposure": zero_blocked,
            "source": "dd_v7_cohort.load (REPL=1, GAPI=14, GAPE=14, ANCHOR=fdate) re-run on runs_k2026 / k2026c"}


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
