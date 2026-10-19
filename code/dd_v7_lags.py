"""Lags between the last located day, the failure record and the first appearance of a replacement in the same slot
(sixth review, 2026-10-19). Written before running. Descriptive. For every failing drive with a complete 30-day
window: L = last day with a located record (dd_v6_runs.py), F = failure date (first failure record, per-drive table),
R = first day another drive is recorded in the same cluster/vault/pod/slot after L (if any within the period).
USAGE  python dd_v7_lags.py <out.json> <runs.parquet> <k.parquet> [...]"""
import sys, os, json, importlib.util
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("r2", os.path.join(HERE, "dd_review2.py"))
r2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(r2)


def q(x):
    x = pd.Series(x).dropna()
    return {"n": int(len(x)), **{f"p{p}": float(np.percentile(x, p)) for p in (5, 25, 50, 75, 95)}} if len(x) else {"n": 0}


def one(rp, kp):
    d = r2.prep(kp); pend = int(d.pend.iloc[0])
    p0 = pd.Timestamp(pd.read_parquet(kp, columns=["period_start"]).period_start.iloc[0])
    fd = d.set_index("serial_number").fdate.dropna(); fd = fd[fd <= pend - 30]
    r = pd.read_parquet(rp); r["vault"] = r.c * 100000 + r.v
    for c in ("t0", "t1"):
        r[c] = (pd.to_datetime(r[c]) - p0).dt.days
    r["f"] = r.s.map(fd)
    r = r[~(r.f < r.t0)]
    last = r[r.f.notna()].sort_values("t1").groupby("s").tail(1).copy()
    last["L"] = np.minimum(last.t1, last.f)
    occ = r[["vault", "p", "sl", "s", "t0"]]
    m = last[["s", "vault", "p", "sl", "L", "f"]].merge(occ, on=["vault", "p", "sl"], suffixes=("", "_o"))
    m = m[(m.s_o != m.s) & (m.t0 > m.L)]
    R = m.groupby("s").t0.min()
    last["R"] = last.s.map(R)
    FL = last.f - last.L; RL = last.R - last.L; RF = last.R - last.f
    return {"n_failures": int(len(last)), "n_no_located_record_before_failure": int(len(fd) - len(last)),
            "F_minus_L": q(FL), "share_F_on_located_day": float((FL == 0).mean()),
            "share_F_le_14_after_L": float(((FL >= 0) & (FL <= 14)).mean()),
            "R_minus_L": q(RL), "R_minus_F": q(RF), "share_with_replacement": float(last.R.notna().mean()),
            "share_replacement_on_or_before_F": float((RF <= 0).mean()),
            "share_replacement_on_or_before_F_among_ext": float((RF[(FL > 0) & (FL <= 14)] <= 0).mean())}


if __name__ == "__main__":
    a = sys.argv[2:]
    res = {os.path.basename(a[i + 1]): one(a[i], a[i + 1]) for i in range(0, len(a), 2)}
    json.dump(res, open(sys.argv[1], "w"), indent=1)
    print(json.dumps(res, indent=1))
