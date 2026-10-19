"""Second-review analyses (2026-10-17). Written before running; all exploratory/sensitivity, not pre-registered.
Input: the corrected-key per-drive tables (cluster, vault, pod, slot). Server key as in the manuscript.

A  Raw counts per set: O, E, placebo mean, unique follow-on drives, follow-on failures counted for >1 index.
B  Absolute 30-day risk per exposed neighbour (index failures with a full 30-day window only):
   exposed = members alive after the index day; risk = failures in days 1-30 / exposed; baseline = E / exposed.
C  Within-server slot placebos (review item 4): keep the real pod and failure times, permute slot labels
   (C1) among all drives of the pod, (C2) among drives of the pod with the same model and age bin.
   Statistic: HR(d1)/HR(d4), normalised by its placebo mean; p = share of placebos >= observed.
D  Index-event filters: FULL (index day <= period end - 30), FIRST (no other failure in the same pod in the
   30 days before the index day), NOAGE20 (drop the open-ended age bin 20 from index and members).
E  Batch proxy aligned to a common date: age at period start = poh0 - 24 * first_day (review: entry date).
F  Influence: per datacenter, leave one datacenter out, drop the 5% of pods with most failures, and the
   largest change from leaving out one vault (main placebo mean held fixed).
G  Items previously on the old key: own slot (distance 0), pod without own slot, per-period near/far,
   near-set time course L1-L5.
Main placebo as in dd_contagion: (pod, slot) permuted jointly within vault x model x age bin.
N placebos per type and seeds fixed below. Vault-clustered bootstrap (1000), placebo mean fixed.
USAGE  python dd_review2.py <out.json> <period.parquet> [...]   (each file is one period; pooled)
"""
import sys, os, json
import numpy as np, pandas as pd
from multiprocessing import Pool

N_PL = int(os.environ.get("NPL", 100))
N_BOOT = 1000
SEED = 20261017
WINS = {"P": (1, 30), "L1": (1, 3), "L2": (4, 7), "L3": (8, 14), "L4": (15, 30), "L5": (31, 60)}
DB = [(0, 0, "d0"), (1, 2, "d1"), (3, 5, "d2"), (6, 15, "d3"), (16, 10 ** 6, "d4")]
SETS = ["POD", "PODx0", "d0", "d1", "d2", "d3", "d4", "d1_sb", "d4_sb", "d1_ob", "d4_ob", "TOME", "REST"]
FILTERS = ["ALL", "FULL", "FIRST", "NOAGE20"]
KEYW = {"ALL": list(WINS), "FULL": ["P"], "FIRST": ["P"], "NOAGE20": ["P"]}


def prep(path):
    d = pd.read_parquet(path).dropna(subset=["vault", "pod", "slot"]).copy()
    p0 = pd.Timestamp(d.period_start.iloc[0]); pend = (pd.Timestamp(d.period_end.iloc[0]) - p0).days
    for c in ("first", "last", "fdate"):
        d[c] = (pd.to_datetime(d[c]) - p0).dt.days
    d["fail"] = d.fail.fillna(0).astype(int); d.loc[d.fail == 0, "fdate"] = np.nan
    d["agebin"] = (d.poh0.fillna(0) // (24 * 182)).clip(upper=20).astype(int)
    d["days"] = d["last"] - d["first"] + 1
    hz = d.groupby(["model", "agebin", "vault"]).agg(f=("fail", "sum"), dd=("days", "sum"))
    d = d.join((hz.f / hz.dd).rename("h"), on=["model", "agebin", "vault"])
    d["stratum"] = d.groupby(["vault", "model", "agebin"]).ngroup()
    d["podkey"] = d.groupby(["vault", "pod"]).ngroup()
    d["podstratum"] = d.groupby(["vault", "pod", "model", "agebin"]).ngroup()
    d["age0"] = d.poh0 - 24 * d["first"]          # E: power-on hours at period start
    d["dc"] = d.datacenter.fillna("na") if "datacenter" in d else "na"
    d["period"] = os.path.basename(path); d["pend"] = pend
    return d.reset_index(drop=True)


def tally(d, pod, slot, detail=False):
    uv = np.unique(d.vault.values); vidx = {v: k for k, v in enumerate(uv)}
    keys = [(f, s, w) for f in FILTERS for s in SETS for w in KEYW[f]]
    O = {k: np.zeros(len(uv)) for k in keys}; E = {k: np.zeros(len(uv)) for k in keys}
    det = {"n_index": {f: 0 for f in FILTERS}}
    if detail:
        det.update({"exposed": {}, "events": {}, "eexp": {}, "followers": {}})
    first, last, fd, h = d["first"].values, d["last"].values, d.fdate.values, d.h.values
    age0, mdl, ab = d.age0.values, d.model.values, d.agebin.values
    pend = d.pend.values[0]
    for v, ix in d.groupby("vault").indices.items():
        k = vidx[v]
        f_, l_, fd_, h_, p_, s_ = first[ix], last[ix], fd[ix], h[ix], pod[ix], slot[ix]
        a_, m_, b_ = age0[ix], mdl[ix], ab[ix]
        fidx = np.where(~np.isnan(fd_))[0]
        for j in fidx:
            t = fd_[j]
            other = np.ones(len(ix), bool); other[j] = False
            sp = p_ == p_[j]; ss = s_ == s_[j]
            dist = np.abs(s_ - s_[j])
            m = {"POD": other & sp, "TOME": other & ss & ~sp, "REST": other & ~sp & ~ss}
            for lo, hi, nm in DB:
                m[nm] = m["POD"] & (dist >= lo) & (dist <= hi)
            m["PODx0"] = m["POD"] & (dist > 0)
            sb = (m_ == m_[j]) & (np.abs(a_ - a_[j]) < 720)
            for nm in ("d1", "d4"):
                m[nm + "_sb"] = m[nm] & sb; m[nm + "_ob"] = m[nm] & ~sb
            prior = sp & other & (fd_ < t) & (fd_ >= t - 30)
            okf = {"ALL": True, "FULL": t <= pend - 30, "FIRST": not prior.any(), "NOAGE20": b_[j] < 20}
            for flt in FILTERS:
                if not okf[flt]:
                    continue
                det["n_index"][flt] += 1
                memb = (b_ < 20) if flt == "NOAGE20" else None
                for w in KEYW[flt]:
                    a, b = WINS[w]
                    st = np.maximum(f_, t + a); en = np.minimum(l_, t + b)
                    dd = np.clip(en - st + 1, 0, None)
                    ev = (fd_ >= t + a) & (fd_ <= t + b)
                    for sn, mm in m.items():
                        if memb is not None:
                            mm = mm & memb
                        E[(flt, sn, w)][k] += float((h_[mm] * dd[mm]).sum())
                        O[(flt, sn, w)][k] += float(ev[mm].sum())
                        if detail and flt == "FULL" and w == "P":
                            alive = mm & (l_ >= t + 1) & ~(fd_ <= t)
                            kk = sn
                            det["exposed"][kk] = det["exposed"].get(kk, 0) + int(alive.sum())
                            det["events"][kk] = det["events"].get(kk, 0) + int((ev & alive).sum())
                            det["eexp"][kk] = det["eexp"].get(kk, 0.0) + float((h_[alive] * dd[alive]).sum())
                        if detail and flt == "ALL" and w == "P":
                            fol = det["followers"].setdefault(sn, {})
                            for q in np.where(ev & mm)[0]:
                                key = (d.period.values[0], int(ix[q]))
                                fol[key] = fol.get(key, 0) + 1
    return O, E, uv, det


def perm_main(d, rng):
    pod = d.pod.values.copy(); slot = d.slot.values.copy()
    for ix in d.groupby("stratum").indices.values():
        if len(ix) > 1:
            p = rng.permutation(ix); pod[ix], slot[ix] = d.pod.values[p], d.slot.values[p]
    return pod, slot


def perm_slot(d, rng, col):
    slot = d.slot.values.copy()
    for ix in d.groupby(col).indices.values():
        if len(ix) > 1:
            slot[ix] = d.slot.values[rng.permutation(ix)]
    return d.pod.values, slot


PERIODS = None


def init(paths):
    global PERIODS
    PERIODS = [prep(p) for p in paths]


def one_placebo(arg):
    kind, i = arg
    rng = np.random.default_rng([SEED, {"main": 0, "C1": 1, "C2": 2}[kind], i])
    out = []
    for d in PERIODS:
        if kind == "main":
            pod, slot = perm_main(d, rng)
        else:
            pod, slot = perm_slot(d, rng, "podkey" if kind == "C1" else "podstratum")
        O, E, uv, _ = tally(d, pod, slot)
        out.append((O, E))
    O = {k: np.concatenate([o[0][k] for o in out]) for k in out[0][0]}
    E = {k: np.concatenate([o[1][k] for o in out]) for k in out[0][1]}
    return kind, O, E


def main(outp, paths):
    init(paths)
    obs = [tally(d, d.pod.values, d.slot.values, detail=True) for d in PERIODS]
    O = {k: np.concatenate([o[0][k] for o in obs]) for k in obs[0][0]}
    E = {k: np.concatenate([o[1][k] for o in obs]) for k in obs[0][1]}
    vault_dc = np.concatenate([d.groupby("vault").dc.first().reindex(o[2]).values for d, o in zip(PERIODS, obs)])
    vault_per = np.concatenate([[d.period.iloc[0]] * len(o[2]) for d, o in zip(PERIODS, obs)])
    podfail = []  # vault-level share of failures in the top 5% pods
    top_mask = []
    for d, o in zip(PERIODS, obs):
        pf = d.groupby("podkey").fail.sum(); thr = pf.quantile(0.95)
        topv = set(d[d.podkey.isin(pf[pf > thr].index)].vault)
        top_mask.append(np.isin(o[2], list(topv)))
    top_mask = np.concatenate(top_mask)
    tasks = [(k, i) for k in ("main", "C1", "C2") for i in range(N_PL)]
    with Pool(4, initializer=init, initargs=(paths,)) as pool:
        pl = {"main": [], "C1": [], "C2": []}
        for kind, Op, Ep in pool.imap_unordered(one_placebo, tasks, chunksize=2):
            pl[kind].append((Op, Ep))
    hr = lambda o, e, sel=slice(None): o[sel].sum() / max(e[sel].sum(), 1e-12)
    rng = np.random.default_rng(SEED)
    nv = len(vault_dc); boots = rng.integers(0, nv, size=(N_BOOT, nv))
    res = {"n_index": {}, "A_raw": {}, "B_absolute": {}, "C_within_server": {}, "D_filters": {}, "E_batch_aligned": {},
           "F_influence": {}, "G_oldkey_items": {}, "mc": {}}
    for f in FILTERS:
        res["n_index"][f] = sum(o[3]["n_index"][f] for o in obs)

    def X(key, sel=None):
        sel = np.ones(nv, bool) if sel is None else sel
        ph = [hr(p[0][key], p[1][key], sel) for p in pl["main"]]; pm = float(np.mean(ph))
        return hr(O[key], E[key], sel) / pm, pm, float(np.std(ph) / np.sqrt(len(ph)))

    def Xci(key):
        x, pm, se = X(key)
        bs = O[key][boots].sum(1) / np.maximum(E[key][boots].sum(1), 1e-12) / pm
        return {"X": x, "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)),
                "O": int(O[key].sum()), "E": float(E[key].sum()), "placebo_mean": pm, "placebo_mc_se": se}

    def ratio(ka, kb, sel=None):
        xa, pa, _ = X(ka, sel); xb, pb, _ = X(kb, sel)
        out = {"ratio": xa / xb, "Oa": int(O[ka][sel].sum() if sel is not None else O[ka].sum()),
               "Ob": int(O[kb][sel].sum() if sel is not None else O[kb].sum())}
        if sel is None:
            ra = O[ka][boots].sum(1) / np.maximum(E[ka][boots].sum(1), 1e-12) / pa
            rb = O[kb][boots].sum(1) / np.maximum(E[kb][boots].sum(1), 1e-12) / pb
            r = ra / np.maximum(rb, 1e-12)
            out.update(lo=float(np.percentile(r, 2.5)), hi=float(np.percentile(r, 97.5)))
        return out

    for s in SETS:
        res["A_raw"][s] = Xci(("ALL", s, "P"))
    fol = {}
    for o in obs:
        for s, dct in o[3]["followers"].items():
            fol.setdefault(s, {}).update(dct)
    for s in ("POD", "d1", "d4"):
        dct = fol.get(s, {})
        res["A_raw"][s].update(unique_followers=len(dct), followers_multi_index=sum(1 for v in dct.values() if v > 1),
                               n_vaults_with_event=int((O[("ALL", s, "P")] > 0).sum()))
    for s in ("POD", "d1", "d2", "d3", "d4", "REST"):
        ex = sum(o[3]["exposed"].get(s, 0) for o in obs); evn = sum(o[3]["events"].get(s, 0) for o in obs)
        ee = sum(o[3]["eexp"].get(s, 0.0) for o in obs)
        res["B_absolute"][s] = {"exposed_drive_events": ex, "failures_30d": evn, "risk_per_10k": 1e4 * evn / max(ex, 1),
                                "baseline_per_10k": 1e4 * ee / max(ex, 1)}
    # C within-server slot placebos
    for kind in ("C1", "C2"):
        ka, kb = ("ALL", "d1", "P"), ("ALL", "d4", "P")
        obs_r = hr(O[ka], E[ka]) / hr(O[kb], E[kb])
        pr = np.array([hr(p[0][ka], p[1][ka]) / max(hr(p[0][kb], p[1][kb]), 1e-12) for p in pl[kind]])
        pm = pr.mean()
        rb = (O[ka][boots].sum(1) / np.maximum(E[ka][boots].sum(1), 1e-12)) / \
             np.maximum(O[kb][boots].sum(1) / np.maximum(E[kb][boots].sum(1), 1e-12), 1e-12) / pm
        res["C_within_server"][kind] = {"obs_ratio_raw": obs_r, "placebo_mean": float(pm), "placebo_sd": float(pr.std()),
                                        "normalised": obs_r / pm, "lo": float(np.percentile(rb, 2.5)),
                                        "hi": float(np.percentile(rb, 97.5)), "p_perm": float((pr >= obs_r).mean()),
                                        "n_placebo": len(pr)}
    for f in ("FULL", "FIRST", "NOAGE20"):
        res["D_filters"][f] = {"POD": Xci((f, "POD", "P")), "d1_d4": ratio((f, "d1", "P"), (f, "d4", "P")),
                               "n_index": res["n_index"][f]}
    res["E_batch_aligned"] = {"same": ratio(("ALL", "d1_sb", "P"), ("ALL", "d4_sb", "P")),
                              "other": ratio(("ALL", "d1_ob", "P"), ("ALL", "d4_ob", "P"))}
    # F influence
    dcs = sorted(set(vault_dc))
    for dc in dcs:
        sel = vault_dc == dc; nsel = ~sel
        res["F_influence"][f"dc_{dc}"] = {"only": ratio(("ALL", "d1", "P"), ("ALL", "d4", "P"), sel),
                                          "without": ratio(("ALL", "d1", "P"), ("ALL", "d4", "P"), nsel),
                                          "n_vaults": int(sel.sum())}
    res["F_influence"]["without_top5pct_pod_vaults"] = ratio(("ALL", "d1", "P"), ("ALL", "d4", "P"), ~top_mask)
    res["F_influence"]["n_vaults_top5pct"] = int(top_mask.sum())
    base = ratio(("ALL", "d1", "P"), ("ALL", "d4", "P"))["ratio"]
    loo = [ratio(("ALL", "d1", "P"), ("ALL", "d4", "P"), np.arange(nv) != i)["ratio"] for i in range(nv)
           if O[("ALL", "d1", "P")][i] + O[("ALL", "d4", "P")][i] > 0]
    res["F_influence"]["leave_one_vault_out"] = {"base": base, "min": float(min(loo)), "max": float(max(loo))}
    # G
    res["G_oldkey_items"]["own_slot_d0"] = Xci(("ALL", "d0", "P"))
    res["G_oldkey_items"]["pod_without_own_slot"] = Xci(("ALL", "PODx0", "P"))
    res["G_oldkey_items"]["d1_timecourse"] = {w: Xci(("ALL", "d1", w)) for w in ("L1", "L2", "L3", "L4", "L5")}
    for per in sorted(set(vault_per)):
        sel = vault_per == per
        res["G_oldkey_items"][f"near_far_{per}"] = ratio(("ALL", "d1", "P"), ("ALL", "d4", "P"), sel)
    res["mc"] = {"d1_placebo_mc_se": X(("ALL", "d1", "P"))[2], "d4_placebo_mc_se": X(("ALL", "d4", "P"))[2],
                 "n_placebo_main": len(pl["main"])}
    json.dump(res, open(outp, "w"), indent=1, default=float)
    print(json.dumps({k: res[k] for k in ("n_index", "C_within_server", "D_filters", "E_batch_aligned")}, indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
