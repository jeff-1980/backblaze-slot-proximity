"""Mechanism test S (SMART event counters of neighbours), EXPLORATORY. Written 2026-10-20 before running; criteria
fixed before any estimate was seen. Extends dd_mechanism.py part B (same pairs, same index failures, same seed for
the far sample) to counters that separate a mechanical from an electrical route:
  mechanical shock / vibration : smart_191 (G-sense error rate), smart_189 (high-fly writes)
  power / backplane / link     : smart_192 (power-off retract count), smart_12 (power cycle count),
                                 smart_199 (UDMA CRC error count), smart_188 (command timeout)
Primary counters: 191 (mechanical) and 192 (electrical). The other four are secondary.

Pairs: every 2024 and 2025 index failure (corrected-key tables k2024/k2025), its NEAR pod-mates (slot distance 1-2)
and up to 4 random FAR pod-mates (distance >= 16), seed 20261015 as in part B. t = index failure date.
Per neighbour and counter, with c(d) = last non-missing raw value on or before day t+d:
  post = 1 if c(7) - c(-1) > 0      (an increment in days 0..7 after the failure)
  pre  = 1 if c(-1) - c(-8) > 0     (an increment in the 7 days before)
  change = post - pre; neighbours without both values for that counter are dropped for that counter.
Statistic DiD = mean(change | NEAR) - mean(change | FAR), vault bootstrap (B = 1000, vault = cluster*100000+vault_raw).
Also reported: the same with a 30-day post window (c(30) - c(-1) versus c(-1) - c(-31)), the share of neighbours with
data for each counter, NEAR and FAR means, and (descriptive) the 1-30-day failure share of NEAR neighbours with versus
without a post increment.
Pre-set reading (7-day window):
  "mechanical route supported"  DiD(191) > 0 with 95% interval lower bound > 0;
  "electrical route supported"  DiD(192) > 0 with 95% interval lower bound > 0;
  both may hold; if neither, "unresolved by SMART". Secondary counters are descriptive.
RUN FIX 2026-10-20 (I/O only, before any estimate was seen): the first PC run hit the duckdb spill cap; the daily
files are now read 15 at a time, which needs no spilling. Results unaffected.
Memory: duckdb memory_limit 6GB, 4 threads, temp under ~/dd/duckdb_tmp (limit 10GB). CPU only. Reads the daily CSVs,
writes only <out.json>.
SMOKE TEST 2026-10-20 (synthetic CSVs, 400 index failures): a planted smart_191 step in 30% of NEAR neighbours is
recovered (DiD 0.28 [0.26, 0.30]); counters without a planted change give DiD 0.
HOLD-OUT 2026H1 (added 2026-10-20 after the 2024-2025 result, before any 2026 SMART data was obtained): the same
statistic and the same pre-set reading on 2026H1 (k2026c.parquet; SMART columns from dd_v16_slim_smart.py parquets).
"Replicated" for the electrical route = DiD(192, 7-day) > 0 with 95% interval lower bound > 0 on 2026H1 alone.
USAGE  python dd_v16_smart.py <out.json> <csv_dir_2024> <csv_dir_2025> <k2024.parquet> <k2025.parquet>
       python dd_v16_smart.py <out.json> <data_dir> [...] --k <k.parquet> [...]   (CSV or slim parquet dirs)
"""
import sys, os, json, glob, importlib.util
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("ddc", os.path.join(HERE, "dd_contagion.py"))
ddc = importlib.util.module_from_spec(spec); spec.loader.exec_module(ddc)
SEED = 20261015
NBOOT = 1000
CNT = ["smart_191_raw", "smart_189_raw", "smart_192_raw", "smart_12_raw", "smart_199_raw", "smart_188_raw"]
PRIMARY = {"mechanical": "smart_191_raw", "electrical": "smart_192_raw"}


def pairs_for(paths):
    rng = np.random.default_rng(SEED)
    pairs = []
    for p in paths:
        d, _ = ddc.prep(p)
        p0 = pd.Timestamp(d.period_start.iloc[0])
        for v, ix in d.groupby("vault").indices.items():
            fd_, p_, s_, ser = d.fdate.values[ix], d.pod.values[ix], d.slot.values[ix], d.serial_number.values[ix]
            for j in np.where(~np.isnan(fd_))[0]:
                mates = (p_ == p_[j]); mates[j] = False
                dist = np.abs(s_ - s_[j])
                near = np.where(mates & (dist >= 1) & (dist <= 2))[0]
                far = np.where(mates & (dist >= 16))[0]
                if len(far) > 4:
                    far = rng.choice(far, 4, replace=False)
                tdate = (p0 + pd.Timedelta(days=int(fd_[j]))).date()
                iid = f"{os.path.basename(p)}|{ser[j]}"
                pairs += [(ser[k], tdate, "NEAR", iid, v, fd_[k] - fd_[j] if not np.isnan(fd_[k]) else np.nan) for k in near]
                pairs += [(ser[k], tdate, "FAR", iid, v, fd_[k] - fd_[j] if not np.isnan(fd_[k]) else np.nan) for k in far]
    return pd.DataFrame(pairs, columns=["serial_number", "t", "role", "iid", "vault", "fail_lag"])


def panel(P, dirs):
    import duckdb
    con = duckdb.connect()
    con.execute("PRAGMA memory_limit='6GB'"); con.execute("PRAGMA threads=4")
    tmp = os.path.expanduser("~/dd/duckdb_tmp"); os.makedirs(tmp, exist_ok=True)
    con.execute(f"PRAGMA temp_directory='{tmp}'"); con.execute("SET max_temp_directory_size='10GB'")
    con.execute("SET preserve_insertion_order=false")
    P = P.copy(); P["t"] = pd.to_datetime(P.t)
    con.register("P", P[["serial_number", "t", "role", "iid", "vault"]])
    files = sorted(sum((glob.glob(os.path.join(d, "**", "*.csv"), recursive=True) for d in dirs), []))
    files = [f for f in files if not os.path.basename(f).startswith("._")]
    pq = sorted(sum((glob.glob(os.path.join(d, "*.parquet")) for d in dirs), []))  # dd_v16_slim_smart.py output
    sel = ", ".join(f"TRY_CAST(r.{c} AS DOUBLE) AS {c}" for c in CNT)
    out = []
    batches = [("csv", files[i:i + 15]) for i in range(0, len(files), 15)] + [("parquet", [f]) for f in pq]
    for kind, fl in batches:  # 15 daily files per query: no spilling (run fix 2026-10-20, see docstring)
        flist = "[" + ", ".join(f"'{f}'" for f in fl) + "]"
        src = (f"read_csv({flist}, union_by_name=true, header=true, all_varchar=true, ignore_errors=true)" if kind == "csv"
               else f"read_parquet({flist}, union_by_name=true)")
        q = f"""
        WITH r AS (SELECT * FROM {src})
        SELECT P.iid, P.role, P.vault, P.serial_number, datediff('day', P.t, CAST(r.date AS DATE)) AS off, {sel}
        FROM r JOIN P ON r.serial_number = P.serial_number
        WHERE CAST(r.date AS DATE) BETWEEN P.t - INTERVAL 31 DAY AND P.t + INTERVAL 30 DAY
        """
        out.append(con.execute(q).df())
    return pd.concat(out, ignore_index=True)



def main(out, dirs, paths):
    rng = np.random.default_rng(SEED)
    P = pairs_for(paths)
    print("pairs", P.role.value_counts().to_dict(), flush=True)
    D = panel(P, dirs).sort_values(["iid", "serial_number", "off"])
    print("panel rows", len(D), flush=True)
    keys = ["iid", "serial_number", "role", "vault"]
    V = {d: D[D.off <= d].groupby(keys, sort=False)[CNT].last() for d in (-31, -8, -1, 7, 30)}  # last non-missing
    per = V[-1][[]].copy()
    for c in CNT:
        a, b, e, a30, b30 = (V[d][c].reindex(per.index) for d in (-8, -1, 7, -31, 30))
        for nm, x, y in (("post7", e, b), ("pre7", b, a), ("post30", b30, b), ("pre30", b, a30)):
            per[f"{c}|{nm}"] = np.where(x.notna() & y.notna(), (x - y > 0).astype(float), np.nan)
    per = per.reset_index().merge(P[["iid", "serial_number", "role", "fail_lag"]].drop_duplicates(["iid", "serial_number", "role"]),
                                  on=["iid", "serial_number", "role"], how="left")
    vaults = np.unique(per.vault.values); vix = np.searchsorted(vaults, per.vault.values)
    W = np.stack([np.bincount(rng.integers(0, len(vaults), len(vaults)), minlength=len(vaults)) for _ in range(NBOOT)]).astype(float)

    def did(post, pre):
        ok = per[post].notna() & per[pre].notna(); ch = (per[post] - per[pre]).values
        out = {}
        sums = {}
        for role in ("NEAR", "FAR"):
            m = (ok & (per.role == role)).values
            sv = np.bincount(vix[m], weights=ch[m], minlength=len(vaults)); nv = np.bincount(vix[m], minlength=len(vaults)).astype(float)
            sums[role] = (sv.sum() / max(nv.sum(), 1), (W @ sv) / np.maximum(W @ nv, 1))
            out[f"n_{role.lower()}"] = int(m.sum())
            out[f"{role.lower()}_post"] = float(per.loc[m, post].mean()); out[f"{role.lower()}_pre"] = float(per.loc[m, pre].mean())
        bs = sums["NEAR"][1] - sums["FAR"][1]
        out.update(DiD=float(sums["NEAR"][0] - sums["FAR"][0]), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)))
        return out

    res = {"meta": {"script": "code/dd_v16_smart.py", "status": "exploratory mechanism test, written before running",
                    "n_pairs": P.role.value_counts().to_dict(), "n_panel_rows": int(len(D)), "n_neighbours_with_rows": int(len(per))},
           "counters": {}}
    for c in CNT:
        res["counters"][c] = {"w7": did(f"{c}|post7", f"{c}|pre7"), "w30": did(f"{c}|post30", f"{c}|pre30")}
        nr = per[per.role == "NEAR"].dropna(subset=[f"{c}|post7"])
        f30 = (nr.fail_lag >= 1) & (nr.fail_lag <= 30)
        res["counters"][c]["near_failshare_by_post7"] = {
            "with_increment": float(f30[nr[f"{c}|post7"] == 1].mean()) if (nr[f"{c}|post7"] == 1).any() else None,
            "without_increment": float(f30[nr[f"{c}|post7"] == 0].mean()) if (nr[f"{c}|post7"] == 0).any() else None,
            "n_with": int((nr[f"{c}|post7"] == 1).sum()), "n_without": int((nr[f"{c}|post7"] == 0).sum())}
        print(c, json.dumps(res["counters"][c]["w7"]), flush=True)
    v = {k: res["counters"][c]["w7"]["lo"] > 0 for k, c in PRIMARY.items()}
    res["verdict"] = {"mechanical_route_supported": bool(v["mechanical"]), "electrical_route_supported": bool(v["electrical"]),
                      "reading": " and ".join(k for k in ("mechanical", "electrical") if v[k]) + " route supported" if any(v.values())
                      else "unresolved by SMART"}
    json.dump(res, open(out, "w"), indent=1, default=float)
    print(json.dumps(res["verdict"]), flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--k" in a:  # dd_v16_smart.py <out.json> <data_dir> [...] --k <k.parquet> [...]
        i = a.index("--k"); main(a[0], a[1:i], a[i + 1:])
    else:
        main(a[0], a[1:3], a[3:5])
