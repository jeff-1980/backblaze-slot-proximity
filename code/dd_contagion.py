"""Backblaze failure-contagion DEEP DIVE (mechanism). Written 2026-10-14, after the round-2 kill test
(kt_backblaze_contagion.py: PASS on 2024 and held-out 2025) and BEFORE any analysis below was run.

DATA AND SPLITS
  Periods (each built into its own per-drive table; windows are truncated at the period end):
    EXPLORATION: 2023H2 (2023-07-01..2023-12-31; pod_slot_num exists from Q3 2023), 2024, 2025.
    CONFIRMATION (held out, not opened until the exploration report is written and this header
    is frozen): every 2026 daily file that Backblaze has published.
  Location: vault = vault_id, pod = (vault_id, pod_id), slot = pod_slot_num.
    tome = (vault_id, slot): the 20 drives sharing an erasure-coded stripe, one per pod.

BASELINE
  Daily hazard h(model, age_bin, vault), age_bin = 6-month bins of smart_9_raw at the drive's first
  appearance in the period (the vault stratum was added after the 2024 diagnostic showed between-vault
  heterogeneity drives the placebo inflation; that diagnostic used 2024 only).

EXPOSURE SETS (disjoint), relative to an index failure of drive i at day t in vault v:
  POD_d   same pod, other drives, split by slot distance |slot_j - slot_i|: d1 = 1-2, d2 = 3-5,
          d3 = 6-15, d4 = 16+ (POD = union).
  TOME    same vault, same slot, other pods.
  REST    same vault, other pod AND other slot.
  Lag windows after t (days): L1 = 1-3, L2 = 4-7, L3 = 8-14, L4 = 15-30, L5 = 31-60. Primary = 1-30.
  For each set: O = failures of members in the window, E = sum h * member drive-days in the window.
  HR = sum O / sum E pooled over periods. "Excess" X = HR / mean(placebo HR).
  Placebo: permute (pod, slot) PAIRS jointly among drives within (vault, model, age_bin), 100 times,
  and recompute every HR. CIs: vault-clustered bootstrap (1000) of the excess, holding the placebo
  mean fixed.
  Sensitivity S1: drop index failures with another failure in the same pod or tome on t-1..t+1.

HYPOTHESES (exploration reports all; confirmation uses these fixed criteria on 2026)
  H1 pod contagion:  X_POD (1-30 d) > 1, bootstrap 95% CI lower bound > 1.
  H2 stripe coupling: X_TOME (1-30 d) > 1, CI lower bound > 1.
  H3 stripe beyond vault: X_TOME / X_REST > 1, CI lower bound > 1.
  H4 proximity (vibration/thermal): within-pod gradient, X_POD_d1 > X_POD_d4, CI of ratio lower > 1.
  H5 rebuild time course: the TOME excess is concentrated early: X_TOME(1-7) / X_TOME(15-60) > 1,
     CI lower > 1 (a rebuild finishes in days; a shared-environment effect would not decay this way).
  On 2026 a hypothesis is CONFIRMED only if its criterion holds; with fewer than 150 index failures in
  2026, results are reported as underpowered and no hypothesis is called confirmed or refuted.

---
ADDENDUM (frozen 2026-10-14 after exploration on 2023H2-2025, BEFORE opening any 2026 file):
  H1-H5 tested on 2026 unchanged (H5 failed in exploration; kept for completeness).
  H6 (post-exploration): TOME excess over days 8-30, X_TOME(8-30) > 1 with CI lower bound > 1.
     Implemented by adding window "M" = (8, 30) to WINS (does not change the RNG stream).
  H7 (batch control, post-exploration, via dd_batch_diag.py): among SAMEBATCH pod pairs only
     (same model and |poh0 diff| < 720 h), X ratio d1/d4 > 1 with CI lower bound > 1.
  Confirmation data = all 2026 daily files published, one period (2026-01-01 .. last available date).
  Same seed, 100 placebos, 1000 vault bootstrap. If 2026 has < 150 index failures: underpowered,
  none of H1-H7 called confirmed or refuted. Paper headline claims = confirmed ones among H1, H4, H6, H7.
---

USAGE
  python dd_contagion.py build <csv_dir> <start YYYY-MM-DD> <end YYYY-MM-DD> <out.parquet>
  python dd_contagion.py analyze <out.json> <period1.parquet> [<period2.parquet> ...]
  Requires duckdb, pandas, numpy, pyarrow.
"""
import sys, json, glob, os
import numpy as np
import pandas as pd

LAGS = {"L1": (1, 3), "L2": (4, 7), "L3": (8, 14), "L4": (15, 30), "L5": (31, 60)}
PRIMARY = (1, 30)
DBINS = [(1, 2, "d1"), (3, 5, "d2"), (6, 15, "d3"), (16, 10 ** 6, "d4")]
N_PLACEBO = 100
N_BOOT = 1000


def build(src, start, end, out):
    import duckdb
    con = duckdb.connect()
    files = sorted(f for f in glob.glob(os.path.join(src, "**", "*.csv"), recursive=True)
                   if start <= os.path.basename(f)[:10] <= end)
    pq = sorted(glob.glob(os.path.join(src, "*.parquet")))  # I/O fix: slim parquet input (see dd_fixes.txt)
    if not files and pq:
        plist = "[" + ", ".join(f"'{f}'" for f in pq) + "]"
        src_sql = f"SELECT * FROM read_parquet({plist}, union_by_name=true) WHERE date >= '{start}' AND date <= '{end}'"
        files = sorted(r[0] for r in duckdb.sql(f"SELECT DISTINCT date FROM ({src_sql})").fetchall())
    else:
        flist = "[" + ", ".join(f"'{f}'" for f in files) + "]"
        src_sql = f"SELECT * FROM read_csv({flist}, union_by_name=true, header=true, all_varchar=true, ignore_errors=true)"
    assert files, "no csv files in range"
    q = f"""
    WITH r AS ({src_sql})
    SELECT serial_number,
           arg_max(model, date) AS model,
           arg_max(TRY_CAST(vault_id AS DOUBLE), date) AS vault,
           arg_max(TRY_CAST(pod_id AS DOUBLE), date) AS pod,
           arg_max(TRY_CAST(pod_slot_num AS DOUBLE), date) AS slot,
           count(DISTINCT pod_id) AS n_pods_seen,
           min(CAST(date AS DATE)) AS first, max(CAST(date AS DATE)) AS last,
           max(TRY_CAST(failure AS INTEGER)) AS fail,
           min(CASE WHEN TRY_CAST(failure AS INTEGER)=1 THEN CAST(date AS DATE) END) AS fdate,
           arg_min(TRY_CAST(smart_9_raw AS DOUBLE), date) AS poh0
    FROM r WHERE vault_id IS NOT NULL AND vault_id <> '' AND pod_id IS NOT NULL AND pod_id <> ''
    GROUP BY serial_number
    """
    d = con.execute(q).df()
    d["period_start"], d["period_end"], d["n_files"] = start, end, len(files)
    d.to_parquet(out, index=False)
    print(json.dumps({"out": out, "n_files": len(files), "n_drives": len(d), "n_fail": int(d.fail.fillna(0).sum()),
                      "slot_missing": int(d.slot.isna().sum()), "moved_pods": int((d.n_pods_seen > 1).sum())}))


def prep(path):
    d = pd.read_parquet(path)
    d = d.dropna(subset=["vault", "pod", "slot"]).copy()
    p0 = pd.Timestamp(d.period_start.iloc[0])
    pend = (pd.Timestamp(d.period_end.iloc[0]) - p0).days
    for c in ("first", "last", "fdate"):
        d[c] = (pd.to_datetime(d[c]) - p0).dt.days
    d["fail"] = d.fail.fillna(0).astype(int)
    d.loc[d.fail == 0, "fdate"] = np.nan
    d["agebin"] = (d.poh0.fillna(0) // (24 * 182)).clip(upper=20).astype(int)
    d["days"] = d["last"] - d["first"] + 1
    hz = d.groupby(["model", "agebin", "vault"]).agg(f=("fail", "sum"), dd=("days", "sum"))
    d = d.join((hz.f / hz.dd).rename("h"), on=["model", "agebin", "vault"])
    d["stratum"] = d.groupby(["vault", "model", "agebin"]).ngroup()
    d["period"] = os.path.basename(path)
    return d.reset_index(drop=True), pend


SETS = ["POD", "POD_d1", "POD_d2", "POD_d3", "POD_d4", "TOME", "REST"]
WINS = {"P": PRIMARY, **LAGS, "E7": (1, 7), "LATE": (15, 60), "M": (8, 30)}


def tally(d, pod, slot, s1):
    """Return dict (set, win) -> per-vault arrays O, E (indexed by vault order) and n_index."""
    vaults = d.vault.values
    uv = np.unique(vaults)
    vidx = {v: k for k, v in enumerate(uv)}
    O = {(s, w): np.zeros(len(uv)) for s in SETS for w in WINS}
    E = {(s, w): np.zeros(len(uv)) for s in SETS for w in WINS}
    nidx = 0
    first, last, fd, h = d["first"].values, d["last"].values, d.fdate.values, d.h.values
    for v, ix in d.groupby("vault").indices.items():
        k = vidx[v]
        f_, l_, fd_, h_, p_, s_ = first[ix], last[ix], fd[ix], h[ix], pod[ix], slot[ix]
        for j in np.where(~np.isnan(fd_))[0]:
            t = fd_[j]
            same_pod = p_ == p_[j]; same_slot = s_ == s_[j]
            other = np.ones(len(ix), bool); other[j] = False
            if s1:
                near = other & (same_pod | same_slot) & (np.abs(fd_ - t) <= 1)
                if near.any():
                    continue
            nidx += 1
            dist = np.abs(s_ - s_[j])
            masks = {"POD": other & same_pod, "TOME": other & same_slot & ~same_pod,
                     "REST": other & ~same_pod & ~same_slot}
            for lo, hi, nm in DBINS:
                masks[f"POD_{nm}"] = masks["POD"] & (dist >= lo) & (dist <= hi)
            for w, (a, b) in WINS.items():
                st = np.maximum(f_, t + a); en = np.minimum(l_, t + b)
                dd = np.clip(en - st + 1, 0, None)
                ev = (fd_ >= t + a) & (fd_ <= t + b)
                for sname, m in masks.items():
                    E[(sname, w)][k] += float((h_[m] * dd[m]).sum())
                    O[(sname, w)][k] += float(ev[m].sum())
    return O, E, nidx, uv


def permuted(d, rng):
    pod = d.pod.values.copy(); slot = d.slot.values.copy()
    for ix in d.groupby("stratum").indices.values():
        if len(ix) > 1:
            p = rng.permutation(ix)
            pod[ix], slot[ix] = d.pod.values[p], d.slot.values[p]
    return pod, slot


def analyze(out, paths):
    rng = np.random.default_rng(20261014)
    periods = [prep(p)[0] for p in paths]
    res = {"periods": [], "main": {}, "S1": {}}
    for s1, key in ((False, "main"), (True, "S1")):
        obs, plc = [], []
        for d in periods:
            obs.append(tally(d, d.pod.values, d.slot.values, s1))
        # pool vault arrays across periods (vault ids may repeat across periods: keep them separate units)
        def pool(tl):
            O = {k: np.concatenate([t[0][k] for t in tl]) for k in tl[0][0]}
            E = {k: np.concatenate([t[1][k] for t in tl]) for k in tl[0][1]}
            return O, E, sum(t[2] for t in tl)
        Oo, Eo, n_index = pool(obs)
        ph = {k: [] for k in Oo}
        n_pl = N_PLACEBO if not s1 else N_PLACEBO // 2
        for _ in range(n_pl):
            tl = [tally(d, *permuted(d, rng), s1) for d in periods]
            Op, Ep, _ = pool(tl)
            for k in Op:
                ph[k].append(Op[k].sum() / max(Ep[k].sum(), 1e-12))
        nv = len(next(iter(Oo.values())))
        boots = rng.integers(0, nv, size=(N_BOOT, nv))
        tab = {}
        for k in Oo:
            hr = Oo[k].sum() / max(Eo[k].sum(), 1e-12)
            pm = float(np.mean(ph[k]))
            bs = Oo[k][boots].sum(1) / np.maximum(Eo[k][boots].sum(1), 1e-12) / pm
            tab[f"{k[0]}|{k[1]}"] = {"O": int(Oo[k].sum()), "E": float(Eo[k].sum()), "HR": float(hr),
                                     "placebo_mean": pm, "placebo_p95": float(np.percentile(ph[k], 95)),
                                     "X": float(hr / pm), "X_lo": float(np.percentile(bs, 2.5)), "X_hi": float(np.percentile(bs, 97.5))}

        def ratio(a, b):
            pa, pb = tab[a]["placebo_mean"], tab[b]["placebo_mean"]
            ka, kb = tuple(a.split("|")), tuple(b.split("|"))
            ra = Oo[ka][boots].sum(1) / np.maximum(Eo[ka][boots].sum(1), 1e-12) / pa
            rb = Oo[kb][boots].sum(1) / np.maximum(Eo[kb][boots].sum(1), 1e-12) / pb
            r = ra / np.maximum(rb, 1e-12)
            return {"ratio": tab[a]["X"] / tab[b]["X"], "lo": float(np.percentile(r, 2.5)), "hi": float(np.percentile(r, 97.5))}
        hyp = {
            "H1_pod": {"X": tab["POD|P"]["X"], "lo": tab["POD|P"]["X_lo"], "pass": tab["POD|P"]["X_lo"] > 1},
            "H2_tome": {"X": tab["TOME|P"]["X"], "lo": tab["TOME|P"]["X_lo"], "pass": tab["TOME|P"]["X_lo"] > 1},
            "H3_tome_vs_rest": ratio("TOME|P", "REST|P"),
            "H4_near_vs_far": ratio("POD_d1|P", "POD_d4|P"),
            "H5_tome_early_vs_late": ratio("TOME|E7", "TOME|LATE"),
            "H6_tome_8_30": {"X": tab["TOME|M"]["X"], "lo": tab["TOME|M"]["X_lo"], "pass": tab["TOME|M"]["X_lo"] > 1},
        }
        for k in ("H3_tome_vs_rest", "H4_near_vs_far", "H5_tome_early_vs_late"):
            hyp[k]["pass"] = hyp[k]["lo"] > 1
        res[key] = {"n_index": n_index, "n_placebo": n_pl, "table": tab, "hypotheses": hyp,
                    "underpowered": n_index < 150}
    res["periods"] = [{"file": d.period.iloc[0], "n_drives": int(len(d)), "n_fail": int(d.fail.sum()),
                       "n_vaults": int(d.vault.nunique()), "slot_max": float(d.slot.max())} for d in periods]
    json.dump(res, open(out, "w"), indent=2, default=lambda o: bool(o) if isinstance(o, np.bool_) else str(o))
    print(json.dumps(res["main"]["hypotheses"], indent=2, default=lambda o: bool(o) if isinstance(o, np.bool_) else str(o)))


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build(*sys.argv[2:6])
    else:
        analyze(sys.argv[2], sys.argv[3:])
