"""Alert cool-down control for the failure-level test (fourth review, 2026-10-18). Written before running.
Question: is the extra recall of the neighbour model (M1) only a result of spreading alerts over more drives?
Same snapshots, features and fitted models as dd_benefit_perfailure.py. In each weekly scan the top q of the
active drives are flagged, but a drive flagged in any of the previous c scans is skipped (its slot in the
budget goes to the next-ranked drive). c = 0 is the original rule. For each model and c we report caught
failures (each failing drive once, flagged in any scan in the 30 days before failure), alerts and distinct
flagged drives. Primary comparison fixed now: M1 with c = 0 against M0 with the smallest c whose distinct
flagged drives reach those of M1 (same inspections per week and at least as many drives inspected), paired
vault bootstrap (1000) of the recall difference, q = 1%. Budgets 0.5% and 2% are reported descriptively.
Amendment 2026-10-18 (after the first run): the first run used an unstable sort, so tied scores were
broken differently from dd_benefit_perfailure.py (pandas rank method="first", i.e. by row order) and c = 0
did not reproduce its counts (M0 96 vs 80). M0 has few distinct scores, so ties at the cut-off matter.
Ties are now broken by row order as in the original, and a tie-break sensitivity (20 random tie orders,
c = 0 and the primary M0 cool-down) is reported.
USAGE  python dd_benefit_cooldown.py <out.json> <train1> ... -- <test.parquet>
"""
import sys, os, json, importlib.util
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("pf", os.path.join(HERE, "dd_benefit_perfailure.py"))
pf = importlib.util.module_from_spec(spec); spec.loader.exec_module(pf)
COOL = (0, 1, 2, 4, 8, 13)


def fit_score(train_paths, test_path):
    rng = np.random.default_rng(pf.SEED)
    tr = pd.concat([pf.snapshots(*pf.ddc.prep(p), rng, True) for p in train_paths], ignore_index=True)
    dt, pend = pf.ddc.prep(test_path); te = pf.snapshots(dt, pend, rng, False)

    def rate(df, keys):
        g = df.assign(yw=df.y * df.w).groupby(keys).agg(yw=("yw", "sum"), w=("w", "sum"))
        return ((g.yw + 0.5) / (g.w + 1)).rename("r")
    glob_r = (tr.y * tr.w).sum() / tr.w.sum()
    r_ma, r_m, r_v = rate(tr, ["model", "agebin"]), rate(tr, ["model"]), rate(tr, ["vault"])
    for df in (tr, te):
        a = df.join(r_ma, on=["model", "agebin"]).r.fillna(df.join(r_m, on="model").r).fillna(glob_r)
        df["lb"] = np.log(a.values); df["lv"] = np.log(df.join(r_v, on="vault").r.fillna(glob_r).values)
    X0 = lambda df: np.column_stack([np.ones(len(df)), df.lb, df.lv])
    X1 = lambda df: np.column_stack([X0(df), df.nb7, df.nb30, df.pd30])
    b0 = pf.logit_fit(X0(tr), tr.y.values.astype(float), tr.w.values)
    b1 = pf.logit_fit(X1(tr), tr.y.values.astype(float), tr.w.values)
    te["s0"], te["s1"] = X0(te) @ b0, X1(te) @ b1
    return te


def flags_cool(te, col, q, c, tie=None):
    taus = np.sort(te.tau.unique()); dids = te.did.values.astype(np.int64)
    last_flag = np.full(dids.max() + 1, -10 ** 6)
    out = np.zeros(len(te), bool); tv = te.tau.values; sv = te[col].values
    for k, tau in enumerate(taus):
        idx = np.where(tv == tau)[0]
        n = int(np.ceil(q * len(idx)))
        ok = np.ones(len(idx), bool) if c == 0 else (k - last_flag[dids[idx]]) > c
        ii = idx[ok]
        order = np.argsort(-sv[ii], kind="stable") if tie is None else np.lexsort((tie[ii], -sv[ii]))
        cand = ii[order][:n]
        out[cand] = True; last_flag[dids[cand]] = k
    return out


def caught(te, fl):
    pos = te.y.values == 1
    c = pd.Series(fl[pos]).groupby(te.did.values[pos]).any()
    return c


def main(out, train_paths, test_path):
    te = fit_score(train_paths, test_path)
    pos = te[te.y == 1]; fail_vault = pos.groupby("did").vault.first()
    rng = np.random.default_rng(pf.SEED + 7)
    res = {"cooldowns": list(COOL), "budgets": {}}
    for q in (0.005, 0.01, 0.02):
        rows = {}
        for col in ("s0", "s1"):
            for c in COOL:
                fl = flags_cool(te, col, q, c); cc = caught(te, fl).reindex(fail_vault.index).fillna(False)
                rows[f"{col}_c{c}"] = {"caught": int(cc.sum()), "recall": float(cc.mean()), "alerts": int(fl.sum()),
                                       "distinct_drives": int(len(np.unique(te.did.values[fl]))), "_c": cc.values}
        m1 = rows["s1_c0"]
        pick = next((c for c in COOL if rows[f"s0_c{c}"]["distinct_drives"] >= m1["distinct_drives"]), COOL[-1])
        a0, a1 = rows[f"s0_c{pick}"]["_c"], m1["_c"]
        uv = np.unique(fail_vault.values); gi = {v: np.where(fail_vault.values == v)[0] for v in uv}
        diffs = []
        for _ in range(1000):
            ix = np.concatenate([gi[v] for v in rng.choice(uv, len(uv))]); diffs.append(a1[ix].mean() - a0[ix].mean())
        for r in rows.values():
            r.pop("_c")
        ties = {}
        for col, c in (("s0", 0), ("s1", 0), ("s0", pick), ("s1", pick)):
            vals = []
            for _ in range(20):
                tie = rng.random(len(te))
                vals.append(int(caught(te, flags_cool(te, col, q, c, tie)).reindex(fail_vault.index).fillna(False).sum()))
            ties[f"{col}_c{c}"] = {"mean": float(np.mean(vals)), "min": int(min(vals)), "max": int(max(vals))}
        n_distinct = {col: int(te[col].round(12).nunique()) for col in ("s0", "s1")}
        res["budgets"][str(q)] = {"rows": rows, "primary": {"M0_cooldown": pick, "diff": float(a1.mean() - a0.mean()),
                                                            "lo": float(np.percentile(diffs, 2.5)), "hi": float(np.percentile(diffs, 97.5))},
                                "tie_random": ties, "n_distinct_scores": n_distinct}
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps({q: v["primary"] for q, v in res["budgets"].items()}, indent=1))
    print(json.dumps(res["budgets"]["0.01"]["rows"], indent=0))


if __name__ == "__main__":
    a = sys.argv[2:]; k = a.index("--")
    main(sys.argv[1], a[:k], a[k + 1])
