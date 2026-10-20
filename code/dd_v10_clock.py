"""Record-clock audit of the near-to-far contrast and its time profile (ninth review, 2026-10-19). Written before running; a
two-draw test run was inspected before the 300-draw run, and nothing was changed after it.

The main analysis dates every failure by its failure record F (first failure record in the per-drive table). In 2025
and 2026H1 the location fields often stop before F, and another drive is sometimes recorded in the slot before F
(results/v7/lags.json). This script asks whether the near-to-far ratio and the short-versus-long contrast depend on
that clock. For every failing drive it takes, from the daily location runs (dd_v6_runs.py):
  L = last day with a located record, capped at F;
  R = first day another drive is recorded in the same (cluster, vault, pod, slot) after L.
Clocks (applied to index failures and follow-on failures alike):
  F   failure record (main analysis);
  L   last located day;
  RM  min(F, R - 1): the day before a replacement was recorded, if that is earlier than F.
Under clocks L and RM a failing drive's exposure also ends on its new event day, and the baseline hazard is
recomputed with the shortened exposure. Everything else follows dd_unified.py / dd_v5_riskset.py: O and E per vault,
sets d1 (slot distance 1-2) and d4 (16 or more) and POD, windows 1-30, 1-3, 31-180 and 61-180 days, placebo of
(pod, slot) within vault x model x age bin (NPL draws), joint vault bootstrap (B = 1000) across periods.
Variant M (matched support, exploration only): only index failures with a complete 180-day follow-up inside their
period (t <= period end - 180), so the short and long windows use the same index failures.
Pair audit (clock F, window 1-30, d1): every counted near pair is classified by the record status of the index and
the follow-on drive: F on a located day (F == L), F after the last located day (F > L), and replacement recorded
on or before F (R <= F).
USAGE  python dd_v10_clock.py <out.json> <runs_dir> <k2023h2c> <k2024> <k2025> <k2026c>
"""
import sys, os, json, importlib.util
import numpy as np, pandas as pd
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("r2", os.path.join(HERE, "dd_review2.py"))
r2 = importlib.util.module_from_spec(spec); spec.loader.exec_module(r2)

NPL = int(os.environ.get("NPL", 100)); B = 1000; SEED = 20261022
WINS = {"P": (1, 30), "S3": (1, 3), "L31": (31, 180), "L61": (61, 180)}
WL = list(WINS); SETS = ["POD", "d1", "d4"]; CLOCKS = ["F", "L", "RM"]; VAR = ["ALL", "M"]
RUNS = {"k2023h2c.parquet": "runs_k2023h2.parquet", "k2024.parquet": "runs_k2024.parquet",
        "k2025.parquet": "runs_k2025.parquet", "k2026c.parquet": "runs_k2026.parquet"}
DATA = None


def lr(runs_path, d, p0):
    """L and R (period days) for every failing drive of d (indexed by serial)."""
    fd = d.set_index("serial_number").fdate.dropna()
    r = pd.read_parquet(runs_path); r["vault"] = r.c * 100000 + r.v
    for c in ("t0", "t1"):
        r[c] = (pd.to_datetime(r[c]) - p0).dt.days
    r["f"] = r.s.map(fd)
    r = r[~(r.f < r.t0)]
    last = r[r.f.notna()].sort_values("t1").groupby("s").tail(1).copy()
    last["L"] = np.minimum(last.t1, last.f)
    occ = r[["vault", "p", "sl", "s", "t0"]]
    m = last[["s", "vault", "p", "sl", "L"]].merge(occ, on=["vault", "p", "sl"], suffixes=("", "_o"))
    m = m[(m.s_o != m.s) & (m.t0 > m.L)]
    last["R"] = last.s.map(m.groupby("s").t0.min())
    return last.set_index("s")[["L", "R"]]


def build(kpath, runs_dir):
    d = r2.prep(kpath)
    p0 = pd.Timestamp(pd.read_parquet(kpath, columns=["period_start"]).period_start.iloc[0])
    t = lr(os.path.join(runs_dir, RUNS[os.path.basename(kpath)]), d, p0)
    d = d.join(t, on="serial_number")
    F = d.fdate.values
    L = np.where(np.isnan(d.L.values), F, d.L.values)
    RM = np.where(np.isnan(d.R.values), F, np.minimum(F, d.R.values - 1))
    out = {}
    for ck, ev in (("F", F), ("L", L), ("RM", RM)):
        e = d[["vault", "pod", "slot", "first", "last", "stratum", "model", "agebin", "pend", "n_pods_seen"]].copy()
        e["fdate"] = ev
        e["last"] = np.where(np.isnan(ev), e["last"], np.minimum(e["last"], ev))
        e["first"] = np.minimum(e["first"], e["last"])
        e["days"] = e["last"] - e["first"] + 1; e["fail"] = (~np.isnan(ev)).astype(int)
        hz = e.groupby(["model", "agebin", "vault"]).agg(f=("fail", "sum"), dd=("days", "sum"))
        e = e.join((hz.f / hz.dd).rename("h"), on=["model", "agebin", "vault"])
        out[ck] = e.reset_index(drop=True)
    st = {"F_eq_L": (F == d.L.values), "F_gt_L": (F > d.L.values), "R_le_F": (d.R.values <= F)}
    return out, {k: np.where(np.isnan(F), False, v) for k, v in st.items()}, d.serial_number.values


def tally(e, pod, slot, audit=None):
    uv = np.unique(e.vault.values); vidx = {v: k for k, v in enumerate(uv)}
    O = np.zeros((len(VAR), len(SETS), len(WL), len(uv))); E = np.zeros_like(O)
    NI = np.zeros((len(VAR), len(WL)))
    first, last, fd, h = e["first"].values, e["last"].values, e.fdate.values, e.h.values
    pend = e.pend.values[0]
    wa = np.array([WINS[w][0] for w in WL]); wb = np.array([WINS[w][1] for w in WL])
    pairs = []
    for v, ix in e.groupby("vault").indices.items():
        k = vidx[v]
        f_, l_, fd_, h_, p_, s_ = first[ix], last[ix], fd[ix], h[ix], pod[ix], slot[ix]
        n = len(ix)
        for j in np.where(~np.isnan(fd_))[0]:
            t = fd_[j]
            other = np.ones(n, bool); other[j] = False
            sp = p_ == p_[j]; dist = np.abs(s_ - s_[j]); pm = other & sp
            M = np.array([pm, pm & (dist >= 1) & (dist <= 2), pm & (dist >= 16)], dtype=float)
            st = np.maximum(f_[None, :], t + wa[:, None]); en = np.minimum(l_[None, :], t + wb[:, None])
            DD = np.clip(en - st + 1, 0, None)
            EV = ((fd_[None, :] >= t + wa[:, None]) & (fd_[None, :] <= t + wb[:, None])).astype(float)
            for vi, vn in enumerate(VAR):
                if vn == "M" and t > pend - 180:
                    continue
                O[vi, :, :, k] += M @ EV.T
                E[vi, :, :, k] += (M * h_[None, :]) @ DD.T
                NI[vi] += (t + wa <= pend)
            if audit is not None:
                for q in np.where((M[1] > 0) & (EV[0] > 0))[0]:
                    pairs.append((ix[j], ix[q], v))
    return O, E, uv, NI, pairs


def init(args):
    global DATA
    DATA = [build(p, args[0])[0] for p in args[1:]]


def one(i):
    rng = np.random.default_rng([SEED, i]); res = []
    for per in DATA:
        pod, slot = r2.perm_main(per["F"].assign(stratum=per["F"].stratum), rng)
        res.append({ck: tuple(x.sum(-1) for x in tally(per[ck], pod, slot)[:2]) for ck in CLOCKS})
    return res


def main(outp, runs_dir, paths):
    built = [build(p, runs_dir) for p in paths]
    obs, audit = [], {}
    for pi, (per, stt, ser) in enumerate(built):
        o = {}
        for ck in CLOCKS:
            O, E, uv, NI, pairs = tally(per[ck], per[ck].pod.values, per[ck].slot.values, audit=(ck == "F"))
            o[ck] = (O, E, uv, NI)
            if ck == "F":
                a = {"n_pairs": len(pairs), "n_vaults": len({p[2] for p in pairs})}
                for who, col in (("index", 0), ("follow", 1)):
                    idx = np.array([p[col] for p in pairs], int)
                    for k, v in stt.items():
                        a[f"{who}_{k}"] = int(v[idx].sum()) if len(idx) else 0
                idx0 = np.array([p[0] for p in pairs], int); idx1 = np.array([p[1] for p in pairs], int)
                a["both_F_eq_L"] = int((stt["F_eq_L"][idx0] & stt["F_eq_L"][idx1]).sum()) if len(pairs) else 0
                a["any_R_le_F"] = int((stt["R_le_F"][idx0] | stt["R_le_F"][idx1]).sum()) if len(pairs) else 0
                audit[os.path.basename(paths[pi])] = a
        obs.append(o)
    global DATA
    DATA = [b[0] for b in built]
    with Pool(4, initializer=init, initargs=([runs_dir] + list(paths),)) as pool:
        pls = list(pool.imap_unordered(one, range(NPL), chunksize=2))
    U = np.unique(np.concatenate([obs[p]["F"][2] for p in range(len(paths))]))
    rng = np.random.default_rng(SEED)
    mult = np.stack([np.bincount(rng.integers(0, len(U), len(U)), minlength=len(U)) for _ in range(B)])
    G = {"explore": [0, 1, 2], "confirm": [3]}
    res = {"audit": audit, "NPL": NPL, "B": B, "groups": {}}
    for g, ps in G.items():
        rg = {}
        for ck in CLOCKS:
            for vi, vn in enumerate(VAR):
                if vn == "M" and g == "confirm":
                    continue
                def x(s, ws, boot):
                    o = e = 0; po = pe = 0
                    for p in ps:
                        O, E, uv, _ = obs[p][ck]; u = np.searchsorted(U, uv)
                        Ov = sum(O[vi, SETS.index(s), WL.index(w)] for w in ws); Ev = sum(E[vi, SETS.index(s), WL.index(w)] for w in ws)
                        if boot:
                            o = o + mult[:, u] @ Ov; e = e + mult[:, u] @ Ev
                        else:
                            o += Ov.sum(); e += Ev.sum()
                        po = po + np.array([sum(pl[p][ck][0][vi, SETS.index(s), WL.index(w)] for w in ws) for pl in pls])
                        pe = pe + np.array([sum(pl[p][ck][1][vi, SETS.index(s), WL.index(w)] for w in ws) for pl in pls])
                    return o / np.maximum(e, 1e-12) / (po / pe).mean()
                ent = {}
                for nm, ws in (("w1_30", ["P"]), ("w1_3", ["S3"]), ("w31_180", ["L31"]), ("w61_180", ["L61"])):
                    pt = x("d1", ws, False) / x("d4", ws, False); bt = x("d1", ws, True) / np.maximum(x("d4", ws, True), 1e-12)
                    n1 = int(sum(obs[p][ck][0][vi, 1, WL.index(ws[0])].sum() for p in ps))
                    n4 = int(sum(obs[p][ck][0][vi, 2, WL.index(ws[0])].sum() for p in ps))
                    nv = int(sum((obs[p][ck][0][vi, 1, WL.index(ws[0])] > 0).sum() for p in ps))
                    ni = int(sum(obs[p][ck][3][vi, WL.index(ws[0])] for p in ps))
                    ent[nm] = {"ratio": float(pt), "lo": float(np.percentile(bt, 2.5)), "hi": float(np.percentile(bt, 97.5)),
                               "near_events": n1, "far_events": n4, "near_vaults": nv, "index_failures": ni, "_b": bt}
                    xp = x("POD", ws, False); bp = x("POD", ws, True)
                    ent[nm]["pod_X"] = float(xp); ent[nm]["pod_lo"] = float(np.percentile(bp, 2.5)); ent[nm]["pod_hi"] = float(np.percentile(bp, 97.5))
                for ln in ("w61_180", "w31_180"):
                    rr = ent["w1_30"]["_b"] / np.maximum(ent[ln]["_b"], 1e-12)
                    p2 = max(2 * min((rr <= 1).mean(), (rr >= 1).mean()), 2 / (B + 1))
                    ent[f"short_over_{ln}"] = {"ratio": ent["w1_30"]["ratio"] / ent[ln]["ratio"], "lo": float(np.percentile(rr, 2.5)),
                                               "hi": float(np.percentile(rr, 97.5)), "p_two_sided": float(min(p2, 1))}
                for nm in ("w1_30", "w1_3", "w31_180", "w61_180"):
                    ent[nm].pop("_b")
                rg[f"{ck}_{vn}"] = ent
        res["groups"][g] = rg
    json.dump(res, open(outp, "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float)[:6000])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
