"""Analysis of the unified run (dd_unified.py). Third review, 2026-10-18. Written before reading the output.
Groups: exploration = 2023H2, 2024, 2025 pooled; confirmation = 2026H1; pooled = all four.
X = (O/E) / mean placebo O/E (main placebo, NPL_MAIN draws). Intervals: vault bootstrap (1000) with the
placebo mean fixed. In pooled analyses the bootstrap resamples vault identities (cluster, vault) jointly
across periods, so the same vault's periods move together (review item 7); heterogeneity is a Wald test on
log near/far ratios with the bootstrap covariance. Monte Carlo SE of each placebo mean is reported.
USAGE  python dd_unified_analyse.py <prefix> <out.json> <period.parquet> [...] (parquets only for n_index_full)
"""
import sys, json
import numpy as np, pandas as pd
from scipy.stats import chi2

B = 1000; SEED = 20261019


def load(prefix):
    a = np.load(prefix + ".npz"); m = json.load(open(prefix + "_meta.json"))
    return a, m


def main(prefix, outp, parquets):
    a, m = load(prefix)
    S, W, AS, KEEP = m["sets"], m["wins"], m["abs_sets"], [tuple(k) for k in m["pl_keep"]]
    P = len(m["periods"]); rng = np.random.default_rng(SEED)
    O = [a[f"O{p}"] for p in range(P)]; E = [a[f"E{p}"] for p in range(P)]
    vid = [np.array(m["periods"][p]["vault"]) for p in range(P)]
    U = np.unique(np.concatenate(vid)); uidx = [np.searchsorted(U, v) for v in vid]
    mult = np.stack([np.bincount(rng.integers(0, len(U), len(U)), minlength=len(U)) for _ in range(B)])  # B x U
    GROUPS = {"explore": [0, 1, 2], "confirm": [3], "pooled": [0, 1, 2, 3]}
    if P == 1:  # test runs on one period
        GROUPS = {"confirm": [0], "pooled": [0]}
    si = lambda s: S.index(s); wi = lambda w: W.index(w)

    def boot_sums(arrs, sel):
        """arrs: list over periods of per-vault vectors; sel: list over periods of bool masks. Returns B sums."""
        tot = np.zeros(B)
        for p, x in enumerate(arrs):
            if x is None:
                continue
            w = mult[:, uidx[p]] * sel[p][None, :]
            tot += w @ x
        return tot

    def pm_total(kind, ps, s, w):
        po = sum(a[f"pl_{kind}_O{p}"][:, si(s), wi(w)] for p in ps); pe = sum(a[f"pl_{kind}_E{p}"][:, si(s), wi(w)] for p in ps)
        r = po / np.maximum(pe, 1e-12)
        return float(r.mean()), float(r.std() / np.sqrt(len(r))), r

    def Xtab(ps, s, w, kind="main"):
        o = sum(O[p][si(s), wi(w)].sum() for p in ps); e = sum(E[p][si(s), wi(w)].sum() for p in ps)
        pm, se, _ = pm_total(kind, ps, s, w)
        sel = [np.ones(len(vid[p]), bool) if p in ps else np.zeros(len(vid[p]), bool) for p in range(P)]
        bo = boot_sums([O[p][si(s), wi(w)] for p in range(P)], sel); be = boot_sums([E[p][si(s), wi(w)] for p in range(P)], sel)
        xb = bo / np.maximum(be, 1e-12) / pm
        return {"O": int(o), "E": float(e), "placebo_mean": pm, "placebo_mc_se": se, "X": float(o / e / pm),
                "X_mc_se": float(o / e / pm * se / pm), "lo": float(np.percentile(xb, 2.5)), "hi": float(np.percentile(xb, 97.5))}, xb

    def ratio(ps, sa, sb, w="P", wb=None, kind="main"):
        wb = wb or w
        ta, xa = Xtab(ps, sa, w, kind); tb, xb = Xtab(ps, sb, wb, kind)
        r = xa / np.maximum(xb, 1e-12)
        return {"ratio": ta["X"] / tb["X"], "lo": float(np.percentile(r, 2.5)), "hi": float(np.percentile(r, 97.5)),
                "Oa": ta["O"], "Ob": tb["O"], "mc_se": float(ta["X"] / tb["X"] * np.hypot(ta["placebo_mc_se"] / ta["placebo_mean"], tb["placebo_mc_se"] / tb["placebo_mean"]))}

    res = {"npl_main": m["npl_main"], "npl_pod": m["npl_pod"], "groups": {}}
    for g, ps in GROUPS.items():
        r = {"tables": {}, "hyp": {}}
        for s in S:
            for w in W:
                r["tables"][f"{s}|{w}"] = Xtab(ps, s, w)[0]
        r["hyp"]["H1_pod"] = r["tables"]["POD|P"]
        r["hyp"]["H2_tome"] = r["tables"]["TOME|P"]
        r["hyp"]["H3_tome_rest"] = ratio(ps, "TOME", "REST")
        r["hyp"]["H4_near_far"] = ratio(ps, "d1", "d4")
        r["hyp"]["H5_tome_early_late"] = ratio(ps, "TOME", "TOME", "E7", "LATE")
        r["hyp"]["H6_tome_8_30"] = r["tables"]["TOME|M"]
        r["hyp"]["H7_samebatch"] = ratio(ps, "d1_sb", "d4_sb")
        r["lag_near_far"] = {w: ratio(ps, "d1", "d4", w) for w in ("L1", "L2", "L3", "L4", "L5", "G1", "G2")}
        # within-server placebos: HR(d1)/HR(d4) observed over placebo mean of the same statistic
        hr = lambda s: sum(O[p][si(s), wi("P")].sum() for p in ps) / sum(E[p][si(s), wi("P")].sum() for p in ps)
        obs = hr("d1") / hr("d4")
        sel = [np.ones(len(vid[p]), bool) if p in ps else np.zeros(len(vid[p]), bool) for p in range(P)]
        bs = {s: boot_sums([O[p][si(s), wi("P")] for p in range(P)], sel) / np.maximum(boot_sums([E[p][si(s), wi("P")] for p in range(P)], sel), 1e-12) for s in ("d1", "d4")}
        r["within_server"] = {}
        for kind in ("C1", "C3"):
            po = {s: sum(a[f"pl_{kind}_O{p}"][:, si(s), wi("P")] for p in ps) for s in ("d1", "d4")}
            pe = {s: sum(a[f"pl_{kind}_E{p}"][:, si(s), wi("P")] for p in ps) for s in ("d1", "d4")}
            pr = (po["d1"] / pe["d1"]) / (po["d4"] / pe["d4"]); pmn = pr.mean()
            rb = bs["d1"] / np.maximum(bs["d4"], 1e-12) / pmn
            r["within_server"][kind] = {"obs_raw": obs, "placebo_mean": float(pmn), "placebo_sd": float(pr.std()),
                                        "normalised": float(obs / pmn), "lo": float(np.percentile(rb, 2.5)),
                                        "hi": float(np.percentile(rb, 97.5)), "p_perm": float((pr >= obs).mean()),
                                        "max_placebo": float(pr.max()), "n": int(len(pr))}
        # absolute risk (index failures with a complete 30-day window)
        r["absolute"] = {}
        for k, s in enumerate(AS):
            ex = [a[f"Aexp{p}"][k] for p in range(P)]; ev = [a[f"Aev{p}"][k] for p in range(P)]; ee = [a[f"Aeexp{p}"][k] for p in range(P)]
            X_, V_, EE = (sum(x.sum() for p, x in enumerate(arr) if p in ps) for arr in (ex, ev, ee))
            rb = boot_sums(ev, sel) / np.maximum(boot_sums(ex, sel), 1e-12)
            dd = [m["periods"][p]["dedup"][s] for p in ps]
            r["absolute"][s] = {"exposed_drive_events": int(X_), "failures_30d": int(V_), "risk_per_10k": 1e4 * V_ / X_,
                                "lo": 1e4 * float(np.percentile(rb, 2.5)), "hi": 1e4 * float(np.percentile(rb, 97.5)),
                                "model_expected_per_10k": 1e4 * EE / X_,
                                "distinct_drives_exposed": int(sum(x[0] for x in dd)), "distinct_drives_failed": int(sum(x[1] for x in dd)),
                                "risk_per_10k_distinct": 1e4 * sum(x[1] for x in dd) / max(sum(x[0] for x in dd), 1)}
        res["groups"][g] = r
    # n index with complete window
    nfull = []
    for f in parquets:
        d = pd.read_parquet(f).dropna(subset=["vault", "pod", "slot"])
        p0 = pd.Timestamp(d.period_start.iloc[0]); pend = (pd.Timestamp(d.period_end.iloc[0]) - p0).days
        fd = (pd.to_datetime(d.fdate[d.fail == 1]) - p0).dt.days
        nfull.append({"file": f, "n_index": int(len(fd)), "n_index_full": int((fd <= pend - 30).sum())})
    res["n_index_full"] = nfull
    # pooled subgroups with per-subgroup placebo means (main), joint vault bootstrap
    meta = {k: np.concatenate([np.array(m["periods"][p][k] if k != "period" else [p] * len(vid[p])) for p in range(P)])
            for k in ("dc", "cluster", "age_y", "period")}
    off = np.cumsum([0] + [len(v) for v in vid])
    kd1, kd4, kpod = KEEP.index(("d1", "P")), KEEP.index(("d4", "P")), KEEP.index(("POD", "P"))

    def sub_est(mask):
        sel = [mask[off[p]:off[p + 1]] for p in range(P)]
        out = {}
        xs = {}
        for nm, s, kk in (("d1", "d1", kd1), ("d4", "d4", kd4), ("POD", "POD", kpod)):
            po = sum((a[f"pl_main_keep{p}"][:, kk, 0, :] * sel[p][None, :]).sum(1) for p in range(P))
            pe = sum((a[f"pl_main_keep{p}"][:, kk, 1, :] * sel[p][None, :]).sum(1) for p in range(P))
            pm = (po / np.maximum(pe, 1e-12)).mean()
            o = sum(O[p][si(s), wi("P")][sel[p]].sum() for p in range(P)); e = sum(E[p][si(s), wi("P")][sel[p]].sum() for p in range(P))
            bo = boot_sums([O[p][si(s), wi("P")] for p in range(P)], sel); be = boot_sums([E[p][si(s), wi("P")] for p in range(P)], sel)
            out[nm] = {"O": int(o), "X": float(o / max(e, 1e-12) / pm)}; xs[nm] = bo / np.maximum(be, 1e-12) / pm
        rb = xs["d1"] / np.maximum(xs["d4"], 1e-12)
        rat = out["d1"]["X"] / out["d4"]["X"] if out["d4"]["X"] > 0 else float("nan")
        return {"ratio": rat, "lo": float(np.percentile(rb, 2.5)), "hi": float(np.percentile(rb, 97.5)),
                "n_d1": out["d1"]["O"], "n_d4": out["d4"]["O"], "pod_X": out["POD"]["X"], "n_vault_periods": int(mask.sum()),
                "n_vaults": int(len(np.unique(np.concatenate([vid[p][sel[p]] for p in range(P)]))))}, np.log(np.maximum(rb, 1e-12))

    def wald(labels, masks):
        ests, logs = [], []
        for lb, mk in zip(labels, masks):
            e, lr = sub_est(mk)
            if e["n_d1"] >= 10 and e["n_d4"] > 0:
                ests.append((lb, e)); logs.append(lr)
        if len(ests) < 2:
            return {"groups": [lb for lb, _ in ests], "note": "fewer than two groups with >= 10 near events"}
        y = np.log([e["ratio"] for _, e in ests]); L = np.array(logs)
        C = np.hstack([-np.ones((len(y) - 1, 1)), np.eye(len(y) - 1)])
        Sg = np.cov(L); cy = C @ y
        Wd = float(cy @ np.linalg.solve(C @ Sg @ C.T, cy))
        return {"groups": [lb for lb, _ in ests], "W": Wd, "df": len(y) - 1, "p": float(chi2.sf(Wd, len(y) - 1))}

    pool = {"all": sub_est(np.ones(len(meta["dc"]), bool))[0], "by": {}, "het": {}}
    agegrp = np.where(meta["age_y"] >= 3, "old", "young")
    cl0 = np.where(meta["cluster"] == 0, "c0", "other")
    fields = {"period": meta["period"].astype(str), "dc": meta["dc"], "cluster": meta["cluster"].astype(str),
              "cluster0": cl0, "age": agegrp, "cl0_age": np.char.add(np.char.add(cl0, "_"), agegrp)}
    for f, v in fields.items():
        labs = sorted(set(v)); pool["by"][f] = {}
        for lb in labs:
            mk = v == lb
            if sum(O[p][si("d1"), wi("P")][mk[off[p]:off[p + 1]]].sum() for p in range(P)) > 0:
                pool["by"][f][lb] = sub_est(mk)[0]
        try:
            pool["het"][f] = wald(labs, [v == lb for lb in labs])
        except Exception as ex:
            pool["het"][f] = {"error": str(ex)}
    pool["leave_out_period"] = {str(p): sub_est(meta["period"] != p)[0] for p in range(P)}
    res["pooled_subgroups"] = pool
    # datacenter x cluster cross table
    xt = []
    for p in range(P):
        df = pd.DataFrame({"dc": m["periods"][p]["dc"], "cluster": m["periods"][p]["cluster"], "npod": m["periods"][p]["npod"],
                           "d1": O[p][si("d1"), wi("P")], "d4": O[p][si("d4"), wi("P")]})
        g = df.groupby(["dc", "cluster"]).agg(vaults=("npod", "size"), pods=("npod", "sum"), near=("d1", "sum"), far=("d4", "sum")).reset_index()
        g["period"] = m["periods"][p]["file"]; xt += g.to_dict("records")
    res["dc_cluster_table"] = xt
    res["exchangeability"] = [m["periods"][p]["exch"] for p in range(P)]
    res["C1_cooccupancy_share"] = m.get("C1_cooccupancy_share")
    json.dump(res, open(outp, "w"), indent=1, default=float)
    for g in GROUPS:
        print(g, json.dumps({k: res["groups"][g]["hyp"][k] for k in ("H1_pod", "H4_near_far")}, default=float))
    print(json.dumps(pool["het"], default=float))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
