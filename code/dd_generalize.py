"""Generalisability of the near/far ratio (second review item 8, jeff chose this direction 2026-10-17).
Written before running; exploratory.
Uses dd_review2.prep/tally/perm_main. Saves per-vault observed and placebo O, E for POD, d1 and d4 (days 1-30)
so that any subgroup can be analysed with its own placebo mean (re-estimated on the subgroup's vaults).
Subgroups: datacenter, cluster, period; leave one datacenter / cluster / period out.
Added 2026-10-17 (after cluster 0 turned out older): vault age group (median power-on years at first
appearance < 3 or >= 3) and its cross with cluster 0 vs other ("c0_young", "c0_old", "oth_young", "oth_old").
Env ONLYMAIN=1 skips the TOPPODS variant.
Heterogeneity: Cochran's Q and I^2 on log near/far ratios, with variances from the vault bootstrap,
for groups with >= 10 near (d1) events.
Variant TOPPODS: drop all drives of the 5% of pods with the highest failure rate (failures per drive-day)
measured in the OTHER periods (so the selection does not use the outcomes being tested; pods seen only in one
period are kept), then rerun. Hazards are kept from the full table. A same-period ranking would select on the
outcome and remove the clustering by construction.
USAGE  python dd_generalize.py <out.json> <period.parquet> [...]
"""
import sys, os, json, importlib.util
import numpy as np
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("r2", os.path.join(HERE, "dd_review2.py"))
r2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(r2)
N_PL = int(os.environ.get("NPL", 100)); N_BOOT = 1000; SEED = 20261018
KEYS = [("ALL", "POD", "P"), ("ALL", "d1", "P"), ("ALL", "d4", "P")]
PER = None


def load(paths, top):
    out = [r2.prep(p) for p in paths]
    if top:
        stats = [d.groupby(["vault", "pod"]).agg(f=("fail", "sum"), dd=("days", "sum")) for d in out]
        for i, d in enumerate(out):
            oth = [s for j, s in enumerate(stats) if j != i]
            if not oth:
                continue
            tot = oth[0].copy()
            for s in oth[1:]:
                tot = tot.add(s, fill_value=0)
            rate = tot.f / tot.dd
            bad = set(rate[rate > rate.quantile(0.95)].index)
            keep = ~np.array([(v, p) in bad for v, p in zip(d.vault.values, d.pod.values)])
            d = d[keep].reset_index(drop=True)
            d["stratum"] = d.groupby(["vault", "model", "agebin"]).ngroup()
            out[i] = d
    return out


def init(paths, top):
    global PER
    PER = load(paths, top)


def run_tally(d, pod, slot):
    O, E, uv, _ = r2.tally(d, pod, slot)
    return {k: O[k] for k in KEYS}, {k: E[k] for k in KEYS}, uv


def one(i):
    rng = np.random.default_rng([SEED, i]); o = []
    for d in PER:
        o.append(run_tally(d, *r2.perm_main(d, rng)))
    return ({k: np.concatenate([x[0][k] for x in o]) for k in KEYS}, {k: np.concatenate([x[1][k] for x in o]) for k in KEYS})


def analyse(paths, top):
    init(paths, top)
    obs = [run_tally(d, d.pod.values, d.slot.values) for d in PER]
    O = {k: np.concatenate([x[0][k] for x in obs]) for k in KEYS}
    E = {k: np.concatenate([x[1][k] for x in obs]) for k in KEYS}
    meta = {"dc": [], "cluster": [], "period": [], "age": [], "cl_age": []}
    for d, x in zip(PER, obs):
        g = d.groupby("vault").agg(dc=("dc", "first"), cl=("cluster", "first"), age=("poh0", "median")).reindex(x[2])
        meta["dc"] += list(g.dc.values); meta["cluster"] += [str(int(c)) for c in g.cl.values]
        ag = np.where(g.age.values / 8766 >= 3, "old", "young")
        meta["age"] += list(ag)
        meta["cl_age"] += [("c0_" if c == 0 else "oth_") + a for c, a in zip(g.cl.values, ag)]
        meta["period"] += [d.period.iloc[0]] * len(x[2])
    meta = {k: np.array(v) for k, v in meta.items()}
    with Pool(4, initializer=init, initargs=(paths, top)) as pool:
        pl = list(pool.imap_unordered(one, range(N_PL)))
    nv = len(meta["dc"]); rng = np.random.default_rng(SEED)

    def est(sel):
        idx = np.where(sel)[0]
        hr = lambda o, e: o.sum() / max(e.sum(), 1e-12)
        pm = {k: np.mean([hr(p[0][k][idx], p[1][k][idx]) for p in pl]) for k in KEYS}
        x = {k: hr(O[k][idx], E[k][idx]) / pm[k] for k in KEYS}
        b = rng.choice(idx, size=(N_BOOT, len(idx)))
        xb = {k: O[k][b].sum(1) / np.maximum(E[k][b].sum(1), 1e-12) / pm[k] for k in KEYS}
        r = xb[KEYS[1]] / np.maximum(xb[KEYS[2]], 1e-12)
        lr = np.log(r[r > 0])
        return {"pod_X": float(x[KEYS[0]]), "ratio": float(x[KEYS[1]] / x[KEYS[2]]),
                "lo": float(np.percentile(r, 2.5)), "hi": float(np.percentile(r, 97.5)),
                "n_d1": int(O[KEYS[1]][idx].sum()), "n_d4": int(O[KEYS[2]][idx].sum()),
                "n_vaults": int(len(idx)), "var_log": float(np.var(lr)) if len(lr) > 10 else None}

    res = {"all": est(np.ones(nv, bool)), "by": {}, "leave_out": {}, "heterogeneity": {}}
    for f in ("dc", "cluster", "period", "age", "cl_age"):
        res["by"][f] = {}; res["leave_out"][f] = {}
        for g in sorted(set(meta[f])):
            sel = meta[f] == g
            if O[KEYS[1]][sel].sum() + O[KEYS[2]][sel].sum() > 0:
                res["by"][f][g] = est(sel)
            res["leave_out"][f][g] = est(~sel)
        grp = [v for v in res["by"][f].values() if v["n_d1"] >= 10 and v["var_log"]]
        if len(grp) >= 2:
            y = np.log([v["ratio"] for v in grp]); w = 1 / np.array([v["var_log"] for v in grp])
            mu = (w * y).sum() / w.sum(); Q = float((w * (y - mu) ** 2).sum()); df = len(grp) - 1
            from math import erf
            res["heterogeneity"][f] = {"k": len(grp), "Q": Q, "df": df, "I2": max(0.0, (Q - df) / Q) if Q > 0 else 0.0,
                                       "pooled_ratio": float(np.exp(mu))}
    return res


def main(outp, paths):
    res = {"main": analyse(paths, False), "n_placebo": N_PL}
    if not os.environ.get("ONLYMAIN"):
        res["drop_top5pct_pods"] = analyse(paths, True)
    try:
        from scipy.stats import chi2
        for v in res.values():
            if isinstance(v, dict):
                for h in v.get("heterogeneity", {}).values():
                    h["p"] = float(chi2.sf(h["Q"], h["df"]))
    except ImportError:
        pass
    json.dump(res, open(outp, "w"), indent=1, default=float)
    print(json.dumps({k: (v["all"], v["heterogeneity"]) if isinstance(v, dict) else v for k, v in res.items()}, indent=1, default=float))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
