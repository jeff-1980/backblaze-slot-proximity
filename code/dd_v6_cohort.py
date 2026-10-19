"""Index-day cohort: absolute 30-day risk and event clustering (fifth review, 2026-10-18). Written before running.
Input: the daily location runs from dd_v6_runs.py (one row per serial and spell of consecutive daily records at
one cluster/vault/pod/slot) and, for the baseline hazard only, the per-drive tables used by every other analysis
(dd_review2.prep: daily hazard per vault x model x six-month age bin).
Index failures: runs that end with failure = 1, on day t, with t <= period end - 30 (complete window).
Cohort: other drives of the same vault with a daily record on day t (a run covering t), not failed on day t,
placed by their slot on day t. Follow-up runs over days t+1 .. t+30 and stops at the end of the run, that is at
the first move (any change of cluster, vault, pod or slot), at the first missing day, or at the last record.
Event: the run ends with failure = 1 within days t+1 .. t+30. Censored: the run ends earlier without failure.
Per set (pod-mates, slot distance 0, 1-2, 3-5, 6-15, >=16, rest of vault) and period group we report
exposures, events, censored exposures, the naive risk (events / exposures), the Kaplan-Meier 30-day cumulative
incidence (censoring at run end), the model expectation (baseline hazard x days followed, per exposure), O/E,
and vault-bootstrap (1000) intervals. The near-to-far contrast is reported as the ratio of O/E (1-2 vs >=16)
and the ratio of Kaplan-Meier risks; these are not placebo-normalised.
Amendment 2026-10-18 (after seeing the 2023H2 runs, before any 2024-2026 runs): 43% of serials have missing
days, almost all on dates when the daily file lists only part of the fleet (e.g. 2023-07-05, 2023-12-18), with
gaps of 1-13 days at an unchanged location. Such gaps are reporting gaps, not exits. Runs of one serial at the
same location separated by at most GAP days (default 14) are therefore merged before the analysis; a change of
location or a longer gap still ends the run. GAP=0 gives the strict version, reported as a sensitivity analysis.
Second amendment (same time): about 6% of failure records are followed by further daily records of the same
drive, so the run does not end on the failure day. The failure date is therefore taken from the per-drive table
(first failure record, as in every other analysis): the run covering that date is cut there and marked failed,
and later records of the drive are dropped. Failing drives often report their last days without location fields,
so their last located run ends a few days before the failure record; if the failure falls at most GAP days after
the end of the drive's last located run, that run is extended to the failure day and marked failed.
USAGE  python dd_v6_cohort.py <out.json> <runs.parquet> <k.parquet> [<runs.parquet> <k.parquet> ...]
       group labels follow the order: the first three pairs = exploration, the fourth = 2026H1.
"""
import sys, os, json, importlib.util
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("r2", os.path.join(HERE, "dd_review2.py"))
r2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(r2)
SETS = ["POD", "d0", "d1", "d2", "d3", "d4", "REST"]
DB = [(0, 0), (1, 2), (3, 5), (6, 15), (16, 10 ** 6)]
W = 30
B = 1000
GAP = int(os.environ.get("GAP", 14))
SEED = 20261021


def load(runs_path, k_path):
    d = r2.prep(k_path)
    p0 = pd.Timestamp(pd.read_parquet(k_path, columns=["period_start"]).period_start.iloc[0])
    pend = int(d.pend.iloc[0])
    h = d.set_index("serial_number").h
    r = pd.read_parquet(runs_path)
    r["vault"] = r.c * 100000 + r.v
    for c in ("t0", "t1"):
        r[c] = (pd.to_datetime(r[c]) - p0).dt.days
    r = merge_gaps(r)
    fd = d.set_index("serial_number").fdate.dropna()
    f = r.s.map(fd)
    r = r[~(f < r.t0)].copy(); f = r.s.map(fd)          # records after the failure day
    cut = f.notna() & (f >= r.t0) & (f <= r.t1)
    last = r.groupby("s").t1.transform("max") == r.t1
    ext = f.notna() & last & (f > r.t1) & (f - r.t1 <= max(GAP, 0))
    cut = cut | ext
    r.loc[cut, "t1"] = f[cut]; r["failed"] = cut.astype(int)
    r["h"] = r.s.map(h)
    r["h"] = r.h.fillna(r.groupby("vault").h.transform("mean")).fillna(h.mean())
    return r.reset_index(drop=True), pend


def merge_gaps(r):
    """Merge consecutive runs of one serial at the same location separated by at most GAP missing days."""
    if GAP <= 0:
        return r
    r = r.sort_values(["s", "t0"]).reset_index(drop=True)
    loc = r[["c", "v", "p", "sl"]].astype(float).values
    same = (r.s.values[1:] == r.s.values[:-1]) & (loc[1:] == loc[:-1]).all(1) \
        & (r.t0.values[1:] - r.t1.values[:-1] - 1 <= GAP) & (r.failed.values[:-1] == 0)
    grp = np.concatenate([[0], np.cumsum(~same)])
    agg = r.groupby(grp).agg(s=("s", "first"), c=("c", "first"), v=("v", "first"), p=("p", "first"),
                             sl=("sl", "first"), vault=("vault", "first"), t0=("t0", "min"), t1=("t1", "max"),
                             n_days=("n_days", "sum"), failed=("failed", "last"))
    return agg.reset_index(drop=True)


def tally(r, pend):
    """Per vault and set: exposures, events, censored, E, removals by day (events and censorings)."""
    out = {}
    s_all = r.s.values
    for v, ix in r.groupby("vault").indices.items():
        t0, t1, fl = r.t0.values[ix], r.t1.values[ix], r.failed.values[ix].astype(bool)
        pod, sl, h, ser = r.p.values[ix], r.sl.values[ix], r.h.values[ix], s_all[ix]
        acc = {s: {"n": 0, "ev": 0, "cens": 0, "E": 0.0, "ek": np.zeros(W + 1), "ck": np.zeros(W + 1)} for s in SETS}
        idx = np.where(fl & (t1 <= pend - W))[0]
        for j in idx:
            t = t1[j]
            mem = (t0 <= t) & (t1 >= t) & (ser != ser[j]) & ~(fl & (t1 == t))
            if not mem.any():
                continue
            days = np.minimum(t1, t + W) - t
            ev = fl & (t1 >= t + 1) & (t1 <= t + W)
            cen = ~ev & (days < W)
            sp = pod == pod[j]; dist = np.abs(sl - sl[j])
            M = {"POD": mem & sp, "REST": mem & ~sp}
            for (lo, hi), nm in zip(DB, ("d0", "d1", "d2", "d3", "d4")):
                M[nm] = M["POD"] & (dist >= lo) & (dist <= hi)
            for s, m in M.items():
                a = acc[s]; a["n"] += m.sum(); a["ev"] += (m & ev).sum(); a["cens"] += (m & cen).sum()
                a["E"] += (h[m] * days[m]).sum()
                np.add.at(a["ek"], days[m & ev], 1); np.add.at(a["ck"], days[m & cen], 1)
        out[v] = acc
    return out


def km(ek, ck, n):
    """30-day cumulative incidence; censorings at day k leave the risk set after the events of day k
    (censorings at day 0 leave before day 1)."""
    at = n - ck[0]; s = 1.0
    for k in range(1, W + 1):
        if at <= 0:
            break
        s *= 1 - ek[k] / at
        at -= ek[k] + ck[k]
    return 1 - s


def summarise(groups, rng):
    res = {}
    for gname, tabs in groups.items():
        keys = [(pi, v) for pi, t in enumerate(tabs) for v in t]
        def agg(sel, s):
            n = sum(tabs[pi][v][s]["n"] for pi, v in sel); ev = sum(tabs[pi][v][s]["ev"] for pi, v in sel)
            ce = sum(tabs[pi][v][s]["cens"] for pi, v in sel); E = sum(tabs[pi][v][s]["E"] for pi, v in sel)
            ek = sum((tabs[pi][v][s]["ek"] for pi, v in sel), np.zeros(W + 1))
            ck = sum((tabs[pi][v][s]["ck"] for pi, v in sel), np.zeros(W + 1))
            return n, ev, ce, E, ek, ck
        # bootstrap over vault identities (all periods of a vault together)
        uv = sorted({v for _, v in keys}); byv = {v: [(pi, vv) for pi, vv in keys if vv == v] for v in uv}
        draws = [[x for v in rng.choice(uv, len(uv)) for x in byv[v]] for _ in range(B)]
        g = {}
        stat = {}
        for s in SETS:
            n, ev, ce, E, ek, ck = agg(keys, s)
            st = {"exposures": int(n), "events": int(ev), "censored": int(ce), "censored_share": ce / max(n, 1),
                  "naive_per_10k": 1e4 * ev / max(n, 1), "km_per_10k": 1e4 * km(ek, ck, n),
                  "expected_per_10k": 1e4 * E / max(n, 1), "O_E": ev / max(E, 1e-12)}
            stat[s] = st
        bs = {s: {"km": [], "oe": [], "naive": []} for s in SETS}
        rr_oe, rr_km = [], []
        for dr in draws:
            tmp = {}
            for s in SETS:
                n, ev, ce, E, ek, ck = agg(dr, s)
                tmp[s] = (1e4 * km(ek, ck, n), ev / max(E, 1e-12), 1e4 * ev / max(n, 1))
                bs[s]["km"].append(tmp[s][0]); bs[s]["oe"].append(tmp[s][1]); bs[s]["naive"].append(tmp[s][2])
            rr_oe.append(tmp["d1"][1] / max(tmp["d4"][1], 1e-12)); rr_km.append(tmp["d1"][0] / max(tmp["d4"][0], 1e-12))
        pc = lambda x: [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]
        for s in SETS:
            stat[s]["km_ci"] = pc(bs[s]["km"]); stat[s]["naive_ci"] = pc(bs[s]["naive"]); stat[s]["O_E_ci"] = pc(bs[s]["oe"])
        g["sets"] = stat
        g["near_far_OE"] = {"ratio": stat["d1"]["O_E"] / stat["d4"]["O_E"], "ci": pc(rr_oe)}
        g["near_far_KM"] = {"ratio": stat["d1"]["km_per_10k"] / stat["d4"]["km_per_10k"], "ci": pc(rr_km)}
        res[gname] = g
    return res


def main(out, pairs):
    rng = np.random.default_rng(SEED)
    tabs, meta = [], []
    for rp, kp in pairs:
        r, pend = load(rp, kp)
        tabs.append(tally(r, pend))
        meta.append({"gap_merge_days": GAP, "runs": rp, "k": kp, "n_runs": int(len(r)), "n_serials": int(r.s.nunique()),
                     "n_index_full": int((r.failed.astype(bool) & (r.t1 <= pend - W)).sum()),
                     "n_failed_runs": int(r.failed.sum()), "pend": pend})
        print("tallied", rp, flush=True)
    groups = {"explore": tabs[:3], "confirm": tabs[3:4]} if len(tabs) >= 4 else {"all": tabs}
    res = {"meta": meta, "groups": summarise(groups, rng)}
    json.dump(res, open(out, "w"), indent=1)
    for gname, g in res["groups"].items():
        print(gname, json.dumps({s: {k: (round(x, 2) if isinstance(x, float) else x) for k, x in v.items()
                                     if k in ("exposures", "events", "censored", "km_per_10k", "expected_per_10k")}
                                 for s, v in g["sets"].items()}), g["near_far_OE"], g["near_far_KM"])


if __name__ == "__main__":
    a = sys.argv[2:]
    main(sys.argv[1], [(a[i], a[i + 1]) for i in range(0, len(a), 2)])
