"""Index-day cohort with an audit of the location reconstruction (sixth review, 2026-10-19). Written before running.
Extends dd_v6_cohort.py. Inputs as there: daily location runs (dd_v6_runs.py) and the per-drive tables
(dd_review2.prep: failure date fdate = first failure record, daily baseline hazard h).

Reconstruction, each step switchable:
  1. Truncation. Runs after a drive's failure date are dropped; a run covering it ends there.
  2. Gap interpolation (GAPI days). Consecutive runs of one drive at one location separated by at most GAPI missing
     days are joined; the joining days are typed "interpolated". With REPL=1 a gap is not joined when another drive's
     run in the same slot starts inside the gap.
  3. End extension (GAPE days). If a drive's failure date falls 1..GAPE days after the end of its last located run,
     that run is extended to the failure date; the added days are typed "extended". With REPL=1 the extension is not
     made when another drive's run in the same slot starts after the run end and on or before the failure date
     (a known replacement); the drive is then censored at its last located day.
  Days inside a located run are typed "observed".
Index failures: drives whose failure date is at least 30 days before the period end and falls on a located day or
at most GAPE days after the end of their last located run (whether or not the extension was blocked); location =
that run's location. Anchor (ANCHOR): "fdate" = the failure date (main), "lastloc" = the last located day.
Cohort: other drives of the vault whose run covers the anchor day t, not failing on day t, placed by that run's slot;
followed over t+1..t+30 to the end of their run. Event: the run ends with the drive's failure date inside the window.
Reported per set (as dd_v6_cohort.py): exposures, events, censored, Kaplan-Meier 30-day risk, naive share,
expected failures over the days followed (O/E), and in addition the model 30-day risk, the mean over exposures of
1 - (1 - h)^30 (the risk if every exposure were followed for the full 30 days), plus exposures and events split by
the type of the anchor day (observed / interpolated / extended) and of the event day; index failures split by type.
The near-to-far contrasts are labelled ratio of Kaplan-Meier risks and (O/E)_near / (O/E)_far.
Groups: exploration (first three periods), 2026H1, and period x cluster (cluster 0 versus others) without bootstrap.
Pre-specified runs (one configuration per call, set by environment):
  main      REPL=1 GAPI=14 GAPE=14 ANCHOR=fdate B=1000
  v6        REPL=0 GAPI=14 GAPE=14 (reproduces dd_v6_cohort.py up to the index rule)
  grid      REPL=1, GAPI in {0,1,3,7,14} with GAPE=14, and GAPE in {0,1,3,7,14} with GAPI=14, B=200
  anchor    REPL=1 GAPI=14 GAPE=14 ANCHOR=lastloc B=1000
No threshold is chosen from these results; the main configuration stays the main configuration.
Bootstrap: B vault resamples; a resample in which a ratio's denominator is zero is dropped and the number kept is
reported (n_valid).
USAGE  python dd_v7_cohort.py <out.json> <runs.parquet> <k.parquet> [... four pairs: 2023H2, 2024, 2025, 2026H1]
"""
import sys, os, json, importlib.util
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("r2", os.path.join(HERE, "dd_review2.py"))
r2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(r2)
SETS = ["POD", "d0", "d1", "d2", "d3", "d4", "REST"]
DB = [(0, 0), (1, 2), (3, 5), (6, 15), (16, 10 ** 6)]
W = 30
GAPI = int(os.environ.get("GAPI", 14)); GAPE = int(os.environ.get("GAPE", 14))
REPL = int(os.environ.get("REPL", 1)); ANCHOR = os.environ.get("ANCHOR", "fdate")
B = int(os.environ.get("B", 1000)); SEED = 20261019
TYP = ("obs", "interp", "ext")


def next_other_start(r):
    """For each run: first start day of a run of another drive in the same slot that starts after this run ends."""
    out = np.full(len(r), 10 ** 9)
    key = r.vault.astype(np.int64).astype(str) + "|" + r.p.astype(str) + "|" + r.sl.astype(str)
    for _, ix in r.groupby(key.values).indices.items():
        if len(ix) < 2:
            continue
        o = ix[np.argsort(r.t0.values[ix])]
        t0, s = r.t0.values[o], r.s.values[o]
        for k, i in enumerate(o):
            j = np.searchsorted(t0, r.t1.values[i], side="right")
            while j < len(o) and s[j] == r.s.values[i]:
                j += 1
            if j < len(o):
                out[i] = t0[j]
    return out


def load(runs_path, k_path):
    d = r2.prep(k_path)
    p0 = pd.Timestamp(pd.read_parquet(k_path, columns=["period_start"]).period_start.iloc[0])
    pend = int(d.pend.iloc[0])
    h = d.set_index("serial_number").h
    fd = d.set_index("serial_number").fdate.dropna()
    r = pd.read_parquet(runs_path)
    r["vault"] = r.c * 100000 + r.v
    for c in ("t0", "t1"):
        r[c] = (pd.to_datetime(r[c]) - p0).dt.days
    r["f"] = r.s.map(fd)
    r = r[~(r.f < r.t0)].copy()                                   # runs after the failure date
    r.loc[r.f.notna() & (r.f <= r.t1), "t1"] = r.f
    r = r.sort_values(["s", "t0"]).reset_index(drop=True)
    r["nos"] = next_other_start(r)
    # segments of observed days, joined by interpolated gaps
    loc = r[["vault", "p", "sl"]].astype(float).values
    gap = r.t0.values[1:] - r.t1.values[:-1] - 1
    join = (r.s.values[1:] == r.s.values[:-1]) & (loc[1:] == loc[:-1]).all(1) & (gap >= 1) & (gap <= GAPI)
    if REPL:
        join &= r.nos.values[:-1] >= r.t0.values[1:]
    rid = np.concatenate([[0], np.cumsum(~join)])
    r["rid"] = rid
    seg = [r[["s", "vault", "p", "sl", "t0", "t1", "rid"]].assign(typ=0)]
    gi = np.where(join)[0]
    seg.append(pd.DataFrame({"s": r.s.values[gi], "vault": r.vault.values[gi], "p": r.p.values[gi], "sl": r.sl.values[gi],
                             "t0": r.t1.values[gi] + 1, "t1": r.t0.values[gi + 1] - 1, "rid": rid[gi], "typ": 1}))
    # extension of each drive's last run to its failure date
    last = r.groupby("s").tail(1)
    lag = last.f - last.t1
    can = last.f.notna() & (lag >= 1) & (lag <= GAPE)
    blocked = can & (last.nos <= last.f) if REPL else can & False
    ex = last[can & ~blocked]
    seg.append(pd.DataFrame({"s": ex.s.values, "vault": ex.vault.values, "p": ex.p.values, "sl": ex.sl.values,
                             "t0": ex.t1.values + 1, "t1": ex.f.values, "rid": ex.rid.values, "typ": 2}))
    sg = pd.concat(seg, ignore_index=True)
    sg[["t0", "t1"]] = sg[["t0", "t1"]].astype(np.int64)
    run_end = sg.groupby("rid").t1.max()
    sg["rend"] = sg.rid.map(run_end)
    sg["f"] = sg.s.map(fd)
    sg["rfail"] = (sg.f == sg.rend).astype(int)
    sg["etyp"] = sg.rid.map(sg.loc[sg.groupby("rid").t1.idxmax()].set_index("rid").typ)   # type of the run's last day
    sg["h"] = sg.s.map(h)
    sg["h"] = sg.h.fillna(sg.groupby("vault").h.transform("mean")).fillna(h.mean())
    # index failures
    li = last[last.f.notna() & (last.f <= pend - W) & (last.f - last.t1 <= GAPE)].copy()
    li["ityp"] = np.where(li.f <= li.t1, 0, np.where(blocked.reindex(li.index, fill_value=False), 3, 2))
    li["t"] = (li.f if ANCHOR == "fdate" else li.t1.clip(upper=li.f)).astype(np.int64)
    meta = {"n_index": int(len(li)), "index_by_type": {k: int((li.ityp == v).sum()) for k, v in (("obs", 0), ("ext", 2), ("ext_blocked", 3))},
            "n_fail_complete": int(((fd <= pend - W)).sum()), "pend": pend,
            "lag_last_located_to_failure": pd.Series((last.f - last.t1)[last.f.notna()]).describe().round(2).to_dict()}
    return sg.reset_index(drop=True), li, meta


def tally(sg, li, pend, clus_of_vault=None):
    out = {}
    sgv = sg.groupby("vault").indices
    for v, ixi in li.groupby("vault").indices.items():
        if v not in sgv:
            continue
        ix = sgv[v]
        a0, a1, typ = sg.t0.values[ix], sg.t1.values[ix], sg.typ.values[ix]
        rend, rfail, f = sg.rend.values[ix], sg.rfail.values[ix].astype(bool), sg.f.values[ix]
        etyp = sg.etyp.values[ix]
        pod, sl, h, ser = sg.p.values[ix], sg.sl.values[ix], sg.h.values[ix], sg.s.values[ix]
        acc = {s: {"n": 0, "ev": 0, "cens": 0, "E": 0.0, "R30": 0.0, "ek": np.zeros(W + 1), "ck": np.zeros(W + 1),
                   "n_typ": [0, 0, 0], "ev_typ": [0, 0, 0]} for s in SETS}
        nidx = [0, 0, 0, 0]
        for j in ixi:
            t, sj, pj, slj = li.t.values[j], li.s.values[j], li.p.values[j], li.sl.values[j]
            mem = (a0 <= t) & (a1 >= t) & (ser != sj) & ~(f == t)
            if ANCHOR != "fdate":
                mem &= ~(f <= t)
            nidx[li.ityp.values[j]] += 1
            if not mem.any():
                continue
            days = np.minimum(rend, t + W) - t
            ev = rfail & (rend >= t + 1) & (rend <= t + W)
            cen = ~ev & (days < W)
            sp = pod == pj; dist = np.abs(sl - slj)
            M = {"POD": mem & sp, "REST": mem & ~sp}
            for (lo, hi), nm in zip(DB, ("d0", "d1", "d2", "d3", "d4")):
                M[nm] = M["POD"] & (dist >= lo) & (dist <= hi)
            for s, m in M.items():
                a = acc[s]; a["n"] += m.sum(); a["ev"] += (m & ev).sum(); a["cens"] += (m & cen).sum()
                a["E"] += (h[m] * days[m]).sum(); a["R30"] += (1 - (1 - h[m]) ** W).sum()
                np.add.at(a["ek"], days[m & ev], 1); np.add.at(a["ck"], days[m & cen], 1)
                for k in range(3):
                    a["n_typ"][k] += (m & (typ == k)).sum()
                    a["ev_typ"][k] += (m & ev & (etyp == k)).sum()
        acc["_nidx"] = nidx
        out[v] = acc
    return out


def km(ek, ck, n):
    at = n - ck[0]; s = 1.0
    for k in range(1, W + 1):
        if at <= 0:
            break
        s *= 1 - ek[k] / at
        at -= ek[k] + ck[k]
    return 1 - s


def agg(tabs, sel, s):
    keys = ("n", "ev", "cens", "E", "R30")
    o = {k: sum(tabs[pi][v][s][k] for pi, v in sel) for k in keys}
    o["ek"] = sum((tabs[pi][v][s]["ek"] for pi, v in sel), np.zeros(W + 1))
    o["ck"] = sum((tabs[pi][v][s]["ck"] for pi, v in sel), np.zeros(W + 1))
    o["n_typ"] = [int(sum(tabs[pi][v][s]["n_typ"][k] for pi, v in sel)) for k in range(3)]
    o["ev_typ"] = [int(sum(tabs[pi][v][s]["ev_typ"][k] for pi, v in sel)) for k in range(3)]
    return o


def stat_of(o):
    n = max(o["n"], 1)
    return {"exposures": int(o["n"]), "events": int(o["ev"]), "censored": int(o["cens"]), "censored_share": o["cens"] / n,
            "naive_per_10k": 1e4 * o["ev"] / n, "km_per_10k": 1e4 * km(o["ek"], o["ck"], o["n"]),
            "expected_per_10k": 1e4 * o["E"] / n, "model30_per_10k": 1e4 * o["R30"] / n, "O_E": o["ev"] / max(o["E"], 1e-12),
            "exposures_by_type": dict(zip(TYP, o["n_typ"])), "events_by_type": dict(zip(TYP, o["ev_typ"]))}


def summarise(tabs, keys, rng, boot=True):
    st = {s: stat_of(agg(tabs, keys, s)) for s in SETS}
    g = {"sets": st, "index_by_type": dict(zip(("obs", "interp", "ext", "ext_blocked"),
                                                [int(sum(tabs[pi][v]["_nidx"][k] for pi, v in keys)) for k in range(4)]))}
    nf = lambda S: (S["d1"]["km_per_10k"] / S["d4"]["km_per_10k"] if S["d4"]["km_per_10k"] > 0 else None,
                    S["d1"]["O_E"] / S["d4"]["O_E"] if S["d4"]["O_E"] > 0 else None)
    rk, ro = nf(st)
    g["near_far_KM"] = {"ratio": rk}; g["near_far_OEratio"] = {"ratio": ro}
    if boot:
        uv = sorted({v for _, v in keys}); byv = {v: [(pi, vv) for pi, vv in keys if vv == v] for v in uv}
        bs = {s: {"km": [], "oe": []} for s in SETS}; rr_k, rr_o = [], []
        for _ in range(B):
            dr = [x for v in rng.choice(uv, len(uv)) for x in byv[v]]
            S = {s: stat_of(agg(tabs, dr, s)) for s in ("d1", "d4", "POD")}
            for s in S:
                bs[s]["km"].append(S[s]["km_per_10k"]); bs[s]["oe"].append(S[s]["O_E"])
            a, b = nf(S)
            if a is not None:
                rr_k.append(a)
            if b is not None:
                rr_o.append(b)
        pc = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))] if len(x) else None
        for s in ("d1", "d4", "POD"):
            st[s]["km_ci"] = pc(bs[s]["km"]); st[s]["O_E_ci"] = pc(bs[s]["oe"])
        g["near_far_KM"].update(ci=pc(rr_k), n_valid=len(rr_k)); g["near_far_OEratio"].update(ci=pc(rr_o), n_valid=len(rr_o))
    return g


def main(out, pairs):
    rng = np.random.default_rng(SEED)
    tabs, meta, clus = [], [], []
    for rp, kp in pairs:
        sg, li, m = load(rp, kp)
        pend = m["pend"]
        tabs.append(tally(sg, li, pend))
        clus.append(sg.groupby("vault").vault.first().floordiv(100000).to_dict())
        m.update(runs=rp, k=kp); meta.append(m)
        print("tallied", rp, m["n_index"], flush=True)
    keys = lambda P: [(pi, v) for pi in P for v in tabs[pi]]
    res = {"config": {"GAPI": GAPI, "GAPE": GAPE, "REPL": REPL, "ANCHOR": ANCHOR, "B": B, "SEED": SEED}, "meta": meta,
           "groups": {"explore": summarise(tabs, keys([0, 1, 2]), rng), "confirm": summarise(tabs, keys([3]), rng)}}
    pc = {}
    for pi, lab in enumerate(("2023H2", "2024", "2025", "2026H1")):
        for cl, nm in ((True, "cluster0"), (False, "other")):
            kk = [(pi, v) for v in tabs[pi] if (int(v) // 100000 == 0) == cl]
            pc[f"{lab}|{nm}"] = summarise(tabs, kk, rng, boot=False) if kk else None
    res["period_cluster"] = pc
    json.dump(res, open(out, "w"), indent=1, default=float)
    for gname, g in res["groups"].items():
        print(gname, {s: (v["exposures"], v["events"], round(v["km_per_10k"], 1), round(v["model30_per_10k"], 1))
                      for s, v in g["sets"].items() if s in ("d0", "d1", "d4")}, g["near_far_KM"], g["near_far_OEratio"], g["index_by_type"])


if __name__ == "__main__":
    a = sys.argv[2:]
    main(sys.argv[1], [(a[i], a[i + 1]) for i in range(0, len(a), 2)])
