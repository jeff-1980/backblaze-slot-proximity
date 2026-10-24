"""Per-failure version of dd_benefit.py (second review, 2026-10-17, item 7). Written before running.
Change: weekly snapshots overlap (30-day label, 7-day step), so dd_benefit counts one failure up to ~4 times.
Here the alert budget is applied per weekly scan (top q of the drives active that week), and a failure is
caught if its drive was flagged in any scan in the 30 days before it failed. Recall = caught failing drives /
failing drives with at least one scan. Vault-clustered bootstrap over drives. Everything else as below.

2026-10-18 addition (third review): per budget, caught failures, alerts, unique alerted drives and
alerts per alerted drive for M0 and M1, and the drives flagged only by M1. Reporting only.

Original header (dd_benefit.py, 2026-10-15):
Engineering value of the neighbour signal (requested by jeff's manuscript review, 2026-10-15).
Written 2026-10-15 BEFORE running on real data. Exploratory; not part of the pre-registered tests.

Question: at a fixed alert budget, does knowing that an adjacent slot (|slot diff| 1-2, same pod)
failed recently improve which drives are flagged, beyond model, age and vault?

Design (per-drive period tables, no SMART: the 2026 slim tables have none):
  Snapshots every 7 days (tau), from day 30 of each period until 30 days before its end. Unit =
  active drive at tau (first <= tau <= last, not failed before tau). Label y = fails in tau+1..tau+30.
  Features: lb = log baseline hazard of (model, age bin) estimated on the TRAINING periods
  (fallback: model, then global), lv = log vault failure rate (training), nb7 / nb30 = an adjacent
  slot failed in tau-7..tau-1 / tau-30..tau-1, pd30 = another slot of the pod (|diff| >= 3) failed in
  tau-30..tau-1.
  Train: 2023H2, 2024, 2025 (negatives subsampled 1 in 10 with weight 10). Test: 2026H1.
  Models: M0 = logistic(lb, lv); M1 = M0 + nb7 + nb30 + pd30.
  Metrics on test: recall of failures among the top q of drive-snapshots by score, q in {0.5%, 1%, 2%};
  AUC; vault-clustered bootstrap (1000) of the recall difference M1 - M0.
  Also: coverage = share of 2026 test failures preceded by an adjacent-slot failure in the past 30 days
  (an upper bound on what the signal can add), and the relative risk of y for nb30 = 1 vs 0 within
  deciles of the M0 score.
READING fixed now: "operational gain" if the 1% budget recall difference has CI lower > 0; otherwise
  the signal is reported as a local risk marker without demonstrated fleet-level gain.

USAGE  python dd_benefit.py <out.json> <train1> <train2> ... -- <test.parquet>
"""
import sys, os, json, importlib.util
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("ddc", os.path.join(HERE, "dd_contagion.py"))
ddc = importlib.util.module_from_spec(spec); spec.loader.exec_module(ddc)
SEED = 20261017
Q = (0.005, 0.01, 0.02)


def snapshots(d, pend, rng, subsample):
    d = d.copy()
    fails = d[~d.fdate.isna()][["vault", "pod", "slot", "fdate"]]
    rows = []
    for tau in range(30, pend - 30 + 1, 7):
        act = d[(d["first"] <= tau) & (d["last"] >= tau) & ~(d.fdate < tau)]
        y = ((act.fdate > tau) & (act.fdate <= tau + 30)).values.astype(np.int8)
        rec = fails[(fails.fdate >= tau - 30) & (fails.fdate <= tau - 1)]
        feats = {}
        for nm, lo, hi, days in (("nb7", 1, 2, 7), ("nb30", 1, 2, 30), ("pd30", 3, 999, 30)):
            r = rec[rec.fdate >= tau - days]
            hit = np.zeros(len(act), bool)
            if len(r):
                m = act[["vault", "pod", "slot"]].reset_index().merge(r[["vault", "pod", "slot"]].rename(columns={"slot": "fs"}), on=["vault", "pod"])
                dd = (m.slot - m.fs).abs()
                ok = m[(dd >= lo) & (dd <= hi)]["index"].unique()
                hit = act.index.isin(ok)
            feats[nm] = hit.astype(np.int8)
        df = pd.DataFrame({"vault": act.vault.values, "model": act.model.values, "agebin": act.agebin.values,
                           "y": y, "tau": tau, "did": act.index.values, **feats})
        if subsample:
            keep = (df.y == 1) | (rng.random(len(df)) < 0.1)
            df = df[keep].copy(); df["w"] = np.where(df.y == 1, 1.0, 10.0)
        else:
            df["w"] = 1.0
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def logit_fit(X, y, w, iters=30):
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ b)))
        W = w * p * (1 - p)
        H = X.T @ (X * W[:, None]) + 1e-6 * np.eye(X.shape[1])
        g = X.T @ (w * (y - p))
        step = np.linalg.solve(H, g); b += step
        if np.abs(step).max() < 1e-8:
            break
    return b


def auc(s, y):
    # FIX 2026-10-24: average ranks for tied scores (argsort gave arbitrary ranks to ties; mock review AUD-03)
    from scipy.stats import rankdata
    r = rankdata(s, method="average")
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def recall_at(s, y, q):
    k = int(np.ceil(q * len(s)))
    top = np.argpartition(-s, k - 1)[:k]
    return y[top].sum() / max(y.sum(), 1)


def main(out, train_paths, test_path):
    rng = np.random.default_rng(SEED)
    tr = []
    for p in train_paths:
        d, pend = ddc.prep(p); tr.append(snapshots(d, pend, rng, True))
    tr = pd.concat(tr, ignore_index=True)
    dt, pend = ddc.prep(test_path); te = snapshots(dt, pend, rng, False)
    # baseline tables from training snapshots (weighted)
    def rate(df, keys):
        g = df.assign(yw=df.y * df.w).groupby(keys).agg(yw=("yw", "sum"), w=("w", "sum"))
        return ((g.yw + 0.5) / (g.w + 1)).rename("r")
    glob_r = (tr.y * tr.w).sum() / tr.w.sum()
    r_ma, r_m, r_v = rate(tr, ["model", "agebin"]), rate(tr, ["model"]), rate(tr, ["vault"])
    for df in (tr, te):
        a = df.join(r_ma, on=["model", "agebin"]).r
        a = a.fillna(df.join(r_m, on="model").r).fillna(glob_r)
        df["lb"] = np.log(a.values)
        df["lv"] = np.log(df.join(r_v, on="vault").r.fillna(glob_r).values)
    X0 = lambda df: np.column_stack([np.ones(len(df)), df.lb, df.lv])
    X1 = lambda df: np.column_stack([X0(df), df.nb7, df.nb30, df.pd30])
    b0 = logit_fit(X0(tr), tr.y.values.astype(float), tr.w.values)
    b1 = logit_fit(X1(tr), tr.y.values.astype(float), tr.w.values)
    s0, s1, y = X0(te) @ b0, X1(te) @ b1, te.y.values
    res = {"n_train_rows": int(len(tr)), "n_test_rows": int(len(te)), "test_positives": int(y.sum()),
           "coef_M1": dict(zip(["const", "lb", "lv", "nb7", "nb30", "pd30"], map(float, b1))),
           "auc": {"M0": auc(s0, y), "M1": auc(s1, y)},
           "coverage_nb30": float(te.nb30[te.y == 1].mean()), "share_rows_nb30": float(te.nb30.mean())}
    te["s0"], te["s1"] = s0, s1
    def flags(col, q):
        r = te.groupby("tau")[col].rank(ascending=False, method="first")
        n = te.groupby("tau")[col].transform("size")
        return (r <= np.ceil(q * n)).values
    pos = te[te.y == 1]
    fail_vault = pos.groupby("did").vault.first()
    rec = {}
    for q in Q:
        f0, f1 = flags("s0", q), flags("s1", q)
        c0 = pd.Series(f0[te.y.values == 1], index=pos.index).groupby(pos.did).any()
        c1 = pd.Series(f1[te.y.values == 1], index=pos.index).groupby(pos.did).any()
        uv = np.unique(fail_vault.values); gi = {v: np.where(fail_vault.values == v)[0] for v in uv}
        a0, a1 = c0.reindex(fail_vault.index).values, c1.reindex(fail_vault.index).values
        diffs = []
        for _ in range(1000):
            ix = np.concatenate([gi[v] for v in rng.choice(uv, len(uv))])
            diffs.append(a1[ix].mean() - a0[ix].mean())
        rec[str(q)] = {"M0": float(a0.mean()), "M1": float(a1.mean()), "diff": float(a1.mean() - a0.mean()),
                       "lo": float(np.percentile(diffs, 2.5)), "hi": float(np.percentile(diffs, 97.5)),
                       "n_failing_drives": int(len(a0))}
        # 2026-10-18 (third review): alert burden, reporting only; recall above is unchanged.
        did = te.did.values
        for nm, ff, cc in (("M0", f0, a0), ("M1", f1, a1)):
            rec[str(q)][nm + "_burden"] = {"caught_failures": int(cc.sum()), "alerts": int(ff.sum()),
                                           "unique_alerted_drives": int(len(np.unique(did[ff]))),
                                           "alerts_per_alerted_drive": float(ff.sum() / max(len(np.unique(did[ff])), 1))}
        new = np.setdiff1d(np.unique(did[f1]), np.unique(did[f0]))
        rec[str(q)]["M1_only"] = {"alerted_drives": int(len(new)), "caught_failures_only_M1": int((a1 & ~a0).sum()),
                                  "lost_failures_only_M0": int((a0 & ~a1).sum())}
    res["recall_per_failure"] = rec
    nbf = pos.groupby("did").nb30.max()
    res["coverage_per_failure_nb30"] = float(nbf.mean())
    dec = pd.qcut(s0, 10, labels=False, duplicates="drop")
    rr = []
    for k in np.unique(dec):
        m = dec == k
        a = y[m & (te.nb30.values == 1)]; b = y[m & (te.nb30.values == 0)]
        rr.append({"decile": int(k), "n_nb": int(len(a)), "rate_nb": float(a.mean()) if len(a) else None, "rate_other": float(b.mean())})
    res["rr_by_baseline_decile"] = rr
    res["reading"] = "OPERATIONAL_GAIN" if rec["0.01"]["lo"] > 0 else "NO_GAIN_DETECTED"
    json.dump(res, open(out, "w"), indent=1, default=float)
    print(json.dumps({k: res[k] for k in ("auc", "recall_per_failure", "coverage_nb30", "coverage_per_failure_nb30", "reading")}, indent=1))


if __name__ == "__main__":
    a = sys.argv[2:]; k = a.index("--")
    main(sys.argv[1], a[:k], a[k + 1])
