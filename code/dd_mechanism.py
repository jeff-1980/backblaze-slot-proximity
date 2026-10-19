"""Mechanism of the confirmed adjacent-slot failure coupling (EXPLORATORY). Written 2026-10-15 before
any mechanism analysis was run. Uses the exploration periods only (2023H2, 2024, 2025); 2026 is not
used (it served as the confirmation set; a future 2026H2 release can confirm mechanism findings).

Notation: index failure i at day t in pod P, slot s_i. NEAR = pod-mates with |slot diff| in {1,2};
FAR = pod-mates with |slot diff| >= 16. X = O/E with the dd_contagion vault-stratified baseline.

A. Hazard-based (uses the per-drive parquets; module dd_contagion.prep for the baseline)
 A1 layout profile: O/E for pod-mates at each exact |slot diff| 1..59 (window 1-30 d), pooled.
    For each candidate row width w in {10, 12, 15, 20, 30}: X(|diff| == w) versus X(|diff| in
    w-4..w-2 or w+2..w+4). Reported descriptively (no pass/fail): a peak at a single w indicates a 2-D
    layout in which the drive directly above/below is also a near neighbour.
 A2 replacement disturbance: a replacement is a drive first seen in the same (vault, pod, slot)
    within 60 days after t; r = its first day. For NEAR neighbours of index failures with a replacement:
    R_rep = X[r, r+6] / X[r-7, r-1] (both windows clipped to >= t+1). Vault-clustered bootstrap CI.
    SUPPORT for "handling/replacement disturbance" if R_rep > 1 with CI lower > 1.
    Also reported: NEAR X (1-30 d) for index failures with vs without a replacement within 60 d.

B. SMART panel (reads the daily CSVs of 2024 and 2025 for the selected drives only)
    Drives: NEAR neighbours of every 2024/2025 index failure, and up to 4 FAR neighbours per index
    (random, seed 20261015). Daily raw values of smart_194 (temperature), smart_5, smart_187,
    smart_188, smart_197, smart_198 for days t-30..t+30.
 B1 thermal: delta_T = mean temp [t+1, t+14] - mean temp [t-14, t-1] per neighbour.
    DiD = mean(delta_T | NEAR) - mean(delta_T | FAR), vault-clustered bootstrap CI.
    SUPPORT for "thermal" if DiD > 0 with CI excluding 0.
    Also: index drive mean temp [t-14, t-1] minus mean temp of its FAR neighbours (descriptive).
 B2 stress/vibration damage in neighbours: new_err = 1 if the sum of the error counters increases
    between t-1 and t+30 (post); pre analogue between t-31 and t-1. DiD of (post - pre) rate,
    NEAR minus FAR, bootstrap CI. SUPPORT for "mechanical stress on neighbours" if DiD > 0, CI excl. 0.
 B3 (descriptive) temperature trajectory: mean temp by day offset -30..+30 for NEAR and FAR.

Interpretation rules fixed now: thermal and stress can both be supported; if none of A2, B1, B2 is
supported the mechanism is reported as unresolved. All results are exploratory.

USAGE
  python dd_mechanism.py A <out.json> <p2023h2.parquet> <p2024.parquet> <p2025.parquet>
  python dd_mechanism.py B <out.json> <csv_dir_2024> <csv_dir_2025> <p2024.parquet> <p2025.parquet>
"""
import sys, os, json, glob, importlib.util
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("ddc", os.path.join(HERE, "dd_contagion.py"))
ddc = importlib.util.module_from_spec(spec); spec.loader.exec_module(ddc)
SEED = 20261015
NBOOT = 1000


def boot_ratio(df, num, den, rng):
    """df: per-vault rows with O_num,E_num,O_den,E_den. Returns ratio of X and CI."""
    g = df.groupby("unit")[[f"O_{num}", f"E_{num}", f"O_{den}", f"E_{den}"]].sum()
    a = g.values
    def r(x):
        xn = x[:, 0].sum() / max(x[:, 1].sum(), 1e-12); xd = x[:, 2].sum() / max(x[:, 3].sum(), 1e-12)
        return xn / max(xd, 1e-12), xn, xd
    est = r(a)
    bs = [r(a[rng.integers(0, len(a), len(a))])[0] for _ in range(NBOOT)]
    return {"ratio": est[0], "X_num": est[1], "X_den": est[2], "lo": float(np.percentile(bs, 2.5)),
            "hi": float(np.percentile(bs, 97.5)), "O_num": float(a[:, 0].sum()), "E_num": float(a[:, 1].sum()),
            "O_den": float(a[:, 2].sum()), "E_den": float(a[:, 3].sum())}


def oe(f_, l_, fd_, h_, m, a, b):
    st = np.maximum(f_[m], a); en = np.minimum(l_[m], b)
    dd = np.clip(en - st + 1, 0, None)
    return float(((fd_[m] >= a) & (fd_[m] <= b)).sum()), float((h_[m] * dd).sum())


def part_A(out, paths):
    rng = np.random.default_rng(SEED)
    prof = np.zeros((60, 2))
    rows = []
    for p in paths:
        d, pend = ddc.prep(p)
        tag = os.path.basename(p)
        for v, ix in d.groupby("vault").indices.items():
            f_, l_, fd_, h_ = d["first"].values[ix], d["last"].values[ix], d.fdate.values[ix], d.h.values[ix]
            p_, s_ = d.pod.values[ix], d.slot.values[ix]
            for j in np.where(~np.isnan(fd_))[0]:
                t = fd_[j]
                mates = (p_ == p_[j]); mates[j] = False
                dist = np.abs(s_ - s_[j])
                for k in range(1, 60):
                    m = mates & (dist == k)
                    if m.any():
                        o, e = oe(f_, l_, fd_, h_, m, t + 1, t + 30)
                        prof[k] += (o, e)
                near = mates & (dist >= 1) & (dist <= 2)
                # replacement: drive first seen in same slot within 60 d after t
                rep = (p_ == p_[j]) & (s_ == s_[j]) & (f_ > t) & (f_ <= t + 60)
                row = {"unit": f"{tag}|{v}", "has_rep": bool(rep.any())}
                o, e = oe(f_, l_, fd_, h_, near, t + 1, t + 30)
                row["O_near30"], row["E_near30"] = o, e
                if rep.any():
                    r = f_[rep].min()
                    o1, e1 = oe(f_, l_, fd_, h_, near, max(r - 7, t + 1), r - 1) if r - 1 >= t + 1 else (0.0, 0.0)
                    o2, e2 = oe(f_, l_, fd_, h_, near, max(r, t + 1), r + 6)
                    row.update(O_pre=o1, E_pre=e1, O_post=o2, E_post=e2, rep_lag=float(r - t))
                rows.append(row)
    R = pd.DataFrame(rows)
    res = {"n_index": int(len(R)), "frac_with_replacement": float(R.has_rep.mean())}
    res["A1_profile"] = {int(k): {"O": prof[k, 0], "E": prof[k, 1], "OE": prof[k, 0] / max(prof[k, 1], 1e-12)} for k in range(1, 60)}
    grid = {}
    for w in (10, 12, 15, 20, 30):
        on = prof[w]; off = sum(prof[k] for k in list(range(w - 4, w - 1)) + list(range(w + 2, w + 5)) if 1 <= k <= 59)
        grid[w] = {"OE_at_w": on[0] / max(on[1], 1e-12), "OE_flank": off[0] / max(off[1], 1e-12), "O_at_w": on[0]}
    res["A1_grid"] = grid
    rep = R[R.has_rep].fillna(0)
    rep_lag = rep.rep_lag.describe().to_dict() if len(rep) else {}
    res["A2_replacement"] = {"n_with_rep": int(len(rep)), "rep_lag_days": rep_lag,
                             **boot_ratio(rep, "post", "pre", rng)} if len(rep) else {}
    R2 = R.copy()
    for g, lab in ((True, "rep"), (False, "norep")):
        R2[f"O_{lab}"] = np.where(R2.has_rep == g, R2.O_near30, 0.0)
        R2[f"E_{lab}"] = np.where(R2.has_rep == g, R2.E_near30, 0.0)
    res["A2_near30_rep_vs_norep"] = boot_ratio(R2, "rep", "norep", rng)
    res["A2_support"] = bool(res.get("A2_replacement", {}).get("lo", 0) > 1)
    json.dump(res, open(out, "w"), indent=2, default=float)
    print(json.dumps({k: v for k, v in res.items() if k != "A1_profile"}, indent=2, default=float))


def part_B(out, dir24, dir25, p24, p25):
    import duckdb
    rng = np.random.default_rng(SEED)
    pairs = []
    for p in (p24, p25):
        d, _ = ddc.prep(p)
        p0 = pd.Timestamp(d.period_start.iloc[0])
        for v, ix in d.groupby("vault").indices.items():
            fd_, p_, s_, ser = d.fdate.values[ix], d.pod.values[ix], d.slot.values[ix], d.serial_number.values[ix]
            for j in np.where(~np.isnan(fd_))[0]:
                mates = (p_ == p_[j]); mates[j] = False
                dist = np.abs(s_ - s_[j])
                near = np.where(mates & (dist <= 2))[0]
                far = np.where(mates & (dist >= 16))[0]
                if len(far) > 4:
                    far = rng.choice(far, 4, replace=False)
                tdate = (p0 + pd.Timedelta(days=int(fd_[j]))).date()
                iid = f"{os.path.basename(p)}|{ser[j]}"
                pairs.append((ser[j], tdate, "INDEX", iid, v))
                pairs += [(ser[k], tdate, "NEAR", iid, v) for k in near]
                pairs += [(ser[k], tdate, "FAR", iid, v) for k in far]
    P = pd.DataFrame(pairs, columns=["serial_number", "t", "role", "iid", "vault"])
    P["t"] = pd.to_datetime(P.t)
    con = duckdb.connect()
    # I/O fix (see dd_fixes.txt): bound memory and spill to WSL ext4 instead of ./.tmp on /mnt/c
    con.execute("PRAGMA memory_limit='8GB'"); con.execute(f"PRAGMA temp_directory='{os.path.expanduser('~/dd/duckdb_tmp')}'")
    con.execute("SET max_temp_directory_size='15GB'")
    con.register("P", P)
    files = sorted(glob.glob(os.path.join(dir24, "**", "*.csv"), recursive=True) + glob.glob(os.path.join(dir25, "**", "*.csv"), recursive=True))
    files = [f for f in files if not os.path.basename(f).startswith("._")]  # I/O fix: skip macOS AppleDouble files (see dd_fixes.txt)
    flist = "[" + ", ".join(f"'{f}'" for f in files) + "]"
    cols = ["smart_194_raw", "smart_5_raw", "smart_187_raw", "smart_188_raw", "smart_197_raw", "smart_198_raw"]
    sel = ", ".join(f"TRY_CAST(r.{c} AS DOUBLE) AS {c}" for c in cols)
    q = f"""
    WITH r AS (SELECT * FROM read_csv({flist}, union_by_name=true, header=true, all_varchar=true, ignore_errors=true))
    SELECT P.iid, P.role, P.vault, P.serial_number, datediff('day', P.t, CAST(r.date AS DATE)) AS off, {sel}
    FROM r JOIN P ON r.serial_number = P.serial_number
    WHERE CAST(r.date AS DATE) BETWEEN P.t - INTERVAL 31 DAY AND P.t + INTERVAL 30 DAY
    """
    D = con.execute(q).df()
    D["err"] = D[["smart_5_raw", "smart_187_raw", "smart_188_raw", "smart_197_raw", "smart_198_raw"]].fillna(0).sum(1)
    res = {"n_index": int((P.role == "INDEX").sum()), "n_near": int((P.role == "NEAR").sum()), "n_far": int((P.role == "FAR").sum()),
           "n_panel_rows": int(len(D))}
    # B3 trajectory
    traj = D[D.role != "INDEX"].groupby(["role", "off"]).smart_194_raw.mean().unstack(0)
    res["B3_temp_traj"] = {r: {int(k): float(v) for k, v in traj[r].dropna().items()} for r in traj.columns}
    # B1
    g = D[D.role != "INDEX"].groupby(["iid", "serial_number", "role", "vault"])
    def win(x, a, b, c):
        return x[(x.off >= a) & (x.off <= b)][c].mean()
    per = g.apply(lambda x: pd.Series({
        "dT": win(x, 1, 14, "smart_194_raw") - win(x, -14, -1, "smart_194_raw"),
        "post_new": float(x[x.off == min(30, x.off.max())].err.sum() - x[x.off == -1].err.sum() > 0) if (x.off == -1).any() else np.nan,
        "pre_new": float(x[x.off == -1].err.sum() - x[x.off == x.off.min()].err.sum() > 0) if (x.off == -1).any() else np.nan,
    })).reset_index()
    def did(col):
        u = per.dropna(subset=[col])
        vaults = u.vault.unique()
        def est(sub):
            return sub[sub.role == "NEAR"][col].mean() - sub[sub.role == "FAR"][col].mean()
        e = est(u)
        byv = {v: u[u.vault == v] for v in vaults}
        bs = []
        for _ in range(NBOOT):
            s = pd.concat([byv[v] for v in rng.choice(vaults, len(vaults))])
            bs.append(est(s))
        return {"DiD": float(e), "lo": float(np.nanpercentile(bs, 2.5)), "hi": float(np.nanpercentile(bs, 97.5)),
                "near_mean": float(u[u.role == "NEAR"][col].mean()), "far_mean": float(u[u.role == "FAR"][col].mean()),
                "n_near": int((u.role == "NEAR").sum()), "n_far": int((u.role == "FAR").sum())}
    res["B1_thermal"] = did("dT")
    res["B1_support"] = bool(res["B1_thermal"]["lo"] > 0)
    per["chg"] = per.post_new - per.pre_new
    res["B2_stress"] = did("chg")
    res["B2_support"] = bool(res["B2_stress"]["lo"] > 0)
    idx = D[D.role == "INDEX"]
    it = idx[(idx.off >= -14) & (idx.off <= -1)].groupby("iid").smart_194_raw.mean()
    ft = D[(D.role == "FAR") & (D.off >= -14) & (D.off <= -1)].groupby("iid").smart_194_raw.mean()
    j = pd.concat([it.rename("i"), ft.rename("f")], axis=1).dropna()
    res["B1_index_minus_far_pre_temp"] = {"mean": float((j.i - j.f).mean()), "n": int(len(j))}
    json.dump(res, open(out, "w"), indent=2, default=float)
    print(json.dumps({k: v for k, v in res.items() if k != "B3_temp_traj"}, indent=2, default=float))


if __name__ == "__main__":
    if sys.argv[1] == "A":
        part_A(sys.argv[2], sys.argv[3:])
    else:
        part_B(*sys.argv[2:7])
