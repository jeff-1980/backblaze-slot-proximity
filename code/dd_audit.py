"""Validity audit of the adjacent-slot failure coupling, requested by jeff's manuscript review (2026-10-15).
Written 2026-10-15 BEFORE running any part below on real data. Robustness / exploratory analyses only;
the pre-registered verdicts (H1, H4, H7) are not re-decided here.

Inputs: the per-drive period tables already used (p2023h2, p2024, p2025 = exploration; p2026 =
confirmation), with the dd_contagion baseline (model x 6-month age bin x vault) and the same O/E
definitions. Window = days 1-30 after the index failure. NEAR = |slot diff| 1-2, FAR = |slot diff| >= 16.

R1 failure-label and location sensitivity. Variants, each with 50 placebo permutations of its own:
   v_fw     drop TOSHIBA MG08ACA16TEY failures dated 2025-07-01..2025-09-30 (Backblaze reported a
            firmware-maintenance episode inflating that model's failure count in 2025 Q3); the drives
            stay in the risk set, their failure is treated as censoring.
   v_podday drop every failure on a pod-day with >= 2 failures (same-day clusters, e.g. bulk removals),
            as index and as follow-on events.
   v_moved  drop drives seen in more than one pod during the period (location ambiguous).
   READING: robust if in every variant the exploration H1 X and H4 ratio keep CI lower bound > 1 and
   the 2026 estimates stay inside the original 2026 95% CI.
R2 hardware class. Pods are classed by slot count in the data (60-slot vs 26-slot servers; no other
   size occurs). NEAR/FAR ratio per class. Gradient "present" in a class if the ratio's CI lower > 1;
   a class with fewer than 30 NEAR follow-on failures is reported as underpowered.
R3 shared-link test (60-slot pods). Classic Backblaze pods connect drives in groups of five through a
   shared SATA link. If the coupling travels along that link, pairs in the same 5-slot block should show
   more excess than pairs at the same slot distance across a block boundary. For block alignment offset
   k in 0..4 (block = (slot + k) // 5) and distance delta in 1..4, standardized ratio
   R_k = O_same / sum_delta(E_same,delta * O_cross,delta / E_cross,delta), vault bootstrap CI.
   READING: shared-link coupling SUPPORTED if R_0 > 1 with CI lower > 1 and R_0 is the largest of the
   five offsets (the slot numbering is assumed to start each block at a multiple of 5; other offsets act
   as negative controls). NOT SUPPORTED otherwise. Decided on exploration; 2026 reported as a check.
   Caveat recorded now: the true slot-to-backplane mapping is not published in the data.
R4 inference. Placebo Monte Carlo SE of the placebo mean, and bootstrap CIs that resample the placebo
   per-vault tallies together with the observed ones (placebo re-estimated within each bootstrap draw).
R5 per-period NEAR/FAR ratio (2023H2, 2024, 2025, 2026) to show consistency over time.

ADDED 2026-10-15 after the first run (bug found: NEAR was coded as |diff| <= 2 and so included
   drives in the failed drive's own slot, i.e. its replacements). Fixed NEAR = |diff| 1-2. Because the
   pre-registered POD set (dd_contagion) also includes same-slot drives, two keys are added: POD_x0
   (pod-mates in other slots) and SLOT0 (later drives in the same slot, mostly replacements). These are
   descriptive; no reading rule.

SMOKE TEST: see the bottom of this file's run log (synthetic data with a planted within-block effect).

USAGE  python dd_audit.py <out.json> <p2023h2> <p2024> <p2025> <p2026>
"""
import sys, os, json, importlib.util
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("ddc", os.path.join(HERE, "dd_contagion.py"))
ddc = importlib.util.module_from_spec(spec); spec.loader.exec_module(ddc)
SEED = 20261016
A, B = 1, 30
N_PL_MAIN, N_PL_VAR, N_BOOT = int(os.environ.get("NPL", 100)), int(os.environ.get("NPLV", 50)), 1000
BASE_KEYS = ["POD", "POD_x0", "SLOT0", "NEAR", "FAR", "REST"]
CLS_KEYS = [f"{k}_{c}" for c in ("C60", "C26") for k in ("NEAR", "FAR")]
BLK_KEYS = [f"B{k}_{sc}_{d}" for k in range(5) for sc in ("same", "cross") for d in range(1, 5)]


def load(path):
    d, _ = ddc.prep(path)
    nslot = d.groupby(["vault", "pod"]).slot.transform("nunique")
    d["cls"] = np.where(nslot >= 40, 60, 26)
    d["fdt"] = pd.Timestamp(d.period_start.iloc[0]) + pd.to_timedelta(d.fdate, unit="D")
    return d


def variant_fd(d, v):
    """Return (fd array used for events, eligible-index mask, drive keep mask)."""
    fd = d.fdate.values.astype(float).copy()
    keep = np.ones(len(d), bool)
    if v == "v_fw":
        m = (d.model.values == "TOSHIBA MG08ACA16TEY") & (d.fdt >= "2025-07-01").values & (d.fdt <= "2025-09-30").values
        fd[m] = np.nan
    elif v == "v_podday":
        f = d[~np.isnan(fd)]
        c = f.groupby(["vault", "pod", "fdate"]).serial_number.transform("size")
        fd[f.index[c.values >= 2]] = np.nan
    elif v == "v_moved":
        keep = d.n_pods_seen.values <= 1
    return fd, keep


def tally(d, pod, slot, fd, keep, full):
    """Per-vault O and E for each key (window 1-30)."""
    keys = BASE_KEYS + (CLS_KEYS + BLK_KEYS if full else [])
    uv = np.unique(d.vault.values); vi = {v: i for i, v in enumerate(uv)}
    O = {k: np.zeros(len(uv)) for k in keys}; E = {k: np.zeros(len(uv)) for k in keys}
    first, last, h, cls = d["first"].values, d["last"].values, d.h.values, d.cls.values
    for v, ix in d.groupby("vault").indices.items():
        ix = ix[keep[ix]]
        if len(ix) == 0:
            continue
        k_ = vi[v]
        f_, l_, fd_, h_, p_, s_, c_ = first[ix], last[ix], fd[ix], h[ix], pod[ix], slot[ix], cls[ix]
        for j in np.where(~np.isnan(fd_))[0]:
            t = fd_[j]
            other = np.ones(len(ix), bool); other[j] = False
            sp = other & (p_ == p_[j]); dist = np.abs(s_ - s_[j])
            st = np.maximum(f_, t + A); en = np.minimum(l_, t + B)
            hd = h_ * np.clip(en - st + 1, 0, None)
            ev = ((fd_ >= t + A) & (fd_ <= t + B)).astype(float)
            m = {"POD": sp, "POD_x0": sp & (dist >= 1), "SLOT0": sp & (dist == 0),
                 "NEAR": sp & (dist >= 1) & (dist <= 2), "FAR": sp & (dist >= 16),
                 "REST": other & (p_ != p_[j]) & (s_ != s_[j])}
            if full:
                cj = "C60" if c_[j] == 60 else "C26"
                m[f"NEAR_{cj}"] = m["NEAR"]; m[f"FAR_{cj}"] = m["FAR"]
                if c_[j] == 60:
                    for kk in range(5):
                        same_blk = ((s_ + kk) // 5) == ((s_[j] + kk) // 5)
                        for dl in range(1, 5):
                            md = sp & (dist == dl)
                            m[f"B{kk}_same_{dl}"] = md & same_blk
                            m[f"B{kk}_cross_{dl}"] = md & ~same_blk
            for key, mk in m.items():
                O[key][k_] += ev[mk].sum(); E[key][k_] += hd[mk].sum()
    return O, E


def run_period(d, rng, fd, keep, n_pl, full):
    obs = tally(d, d.pod.values, d.slot.values, fd, keep, full)
    pl = []
    for _ in range(n_pl):
        p, s = ddc.permuted(d, rng)
        pl.append(tally(d, p, s, fd, keep, False))
    return obs, pl


def pool(results):
    """results: list over periods of (obs, pl). Returns observed O,E (concatenated vault units) and placebo arrays."""
    O = {k: np.concatenate([r[0][0][k] for r in results]) for k in results[0][0][0]}
    E = {k: np.concatenate([r[0][1][k] for r in results]) for k in results[0][0][1]}
    n_pl = len(results[0][1])
    PO = {k: np.stack([np.concatenate([r[1][i][0][k] for r in results]) for i in range(n_pl)]) for k in BASE_KEYS}
    PE = {k: np.stack([np.concatenate([r[1][i][1][k] for r in results]) for i in range(n_pl)]) for k in BASE_KEYS}
    return O, E, PO, PE


def summarize(O, E, PO, PE, rng, full):
    nv = len(O["POD"])
    C = np.zeros((N_BOOT, nv))
    idx = rng.integers(0, nv, size=(N_BOOT, nv))
    for b in range(N_BOOT):
        np.add.at(C[b], idx[b], 1)
    res = {}
    X, Xb, Xb_fix = {}, {}, {}
    for k in BASE_KEYS:
        hr = O[k].sum() / E[k].sum()
        phr = PO[k].sum(1) / PE[k].sum(1)
        pm = phr.mean()
        hrb = (C @ O[k]) / (C @ E[k])
        phrb = (PO[k] @ C.T) / (PE[k] @ C.T)       # n_pl x N_BOOT
        X[k] = hr / pm; Xb[k] = hrb / phrb.mean(0); Xb_fix[k] = hrb / pm
        res[k] = {"O": float(O[k].sum()), "X": X[k], "placebo_mean": float(pm),
                  "placebo_mc_se": float(phr.std(ddof=1) / np.sqrt(len(phr))),
                  "ci_fixed_placebo": [float(np.percentile(Xb_fix[k], 2.5)), float(np.percentile(Xb_fix[k], 97.5))],
                  "ci_reestimated_placebo": [float(np.percentile(Xb[k], 2.5)), float(np.percentile(Xb[k], 97.5))]}
    for nm, a, b in (("near_far", "NEAR", "FAR"),):
        r = Xb[a] / Xb[b]; rf = Xb_fix[a] / Xb_fix[b]
        res[nm] = {"ratio": X[a] / X[b], "ci_fixed_placebo": [float(np.percentile(rf, 2.5)), float(np.percentile(rf, 97.5))],
                   "ci_reestimated_placebo": [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]}
    if full:
        def raw_ratio(a, b):   # ratio of O/E (placebo means of NEAR and FAR cancel approximately; reported as such)
            est = (O[a].sum() / E[a].sum()) / (O[b].sum() / E[b].sum())
            bb = ((C @ O[a]) / (C @ E[a])) / ((C @ O[b]) / (C @ E[b]))
            return {"ratio": float(est), "lo": float(np.nanpercentile(bb, 2.5)), "hi": float(np.nanpercentile(bb, 97.5)),
                    "O_near": float(O[a].sum()), "O_far": float(O[b].sum())}
        res["R2_class"] = {c: raw_ratio(f"NEAR_{c}", f"FAR_{c}") for c in ("C60", "C26")}
        for c in ("C60", "C26"):
            res["R2_class"][c]["underpowered"] = res["R2_class"][c]["O_near"] < 30
        blk = {}
        for kk in range(5):
            def R(Ov, Ev):   # Ov/Ev: dict delta -> (same, cross) arrays already reduced
                num = sum(Ov[dl][0] for dl in range(1, 5))
                den = sum(Ev[dl][0] * Ov[dl][1] / max(Ev[dl][1], 1e-12) for dl in range(1, 5))
                return num / max(den, 1e-12)
            Ov = {dl: (O[f"B{kk}_same_{dl}"].sum(), O[f"B{kk}_cross_{dl}"].sum()) for dl in range(1, 5)}
            Ev = {dl: (E[f"B{kk}_same_{dl}"].sum(), E[f"B{kk}_cross_{dl}"].sum()) for dl in range(1, 5)}
            est = R(Ov, Ev)
            bs = []
            for b in range(N_BOOT):
                Ovb = {dl: (C[b] @ O[f"B{kk}_same_{dl}"], C[b] @ O[f"B{kk}_cross_{dl}"]) for dl in range(1, 5)}
                Evb = {dl: (C[b] @ E[f"B{kk}_same_{dl}"], C[b] @ E[f"B{kk}_cross_{dl}"]) for dl in range(1, 5)}
                bs.append(R(Ovb, Evb))
            blk[f"offset{kk}"] = {"R": float(est), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)),
                                  "O_same": float(sum(v[0] for v in Ov.values())), "O_cross": float(sum(v[1] for v in Ov.values()))}
        r0 = blk["offset0"]
        blk["reading"] = "SUPPORTED" if (r0["lo"] > 1 and r0["R"] == max(v["R"] for k, v in blk.items() if k.startswith("offset"))) else "NOT_SUPPORTED"
        res["R3_block"] = blk
    return res


def main(out, paths):
    rng = np.random.default_rng(SEED)
    ds = [load(p) for p in paths]
    names = [os.path.basename(p) for p in paths]
    groups = {"exploration": [i for i, n in enumerate(names) if "2026" not in n],
              "confirmation_2026": [i for i, n in enumerate(names) if "2026" in n]}
    res = {"periods": names, "pod_class_counts": {}, "variants": {}, "per_period": {}}
    for d, n in zip(ds, names):
        pc = d.groupby(["vault", "pod"]).cls.first().value_counts()
        res["pod_class_counts"][n] = {str(k): int(v) for k, v in pc.items()}
    for v in ("base", "v_fw", "v_podday", "v_moved"):
        full = v == "base"
        n_pl = N_PL_MAIN if full else N_PL_VAR
        per = []
        for d in ds:
            fd, keep = variant_fd(d, v)
            per.append(run_period(d, rng, fd, keep, n_pl, full))
            print(v, d.period.iloc[0], "done", flush=True)
        res["variants"][v] = {}
        for g, ix in groups.items():
            if ix:
                res["variants"][v][g] = summarize(*pool([per[i] for i in ix]), rng, full)
        if full:
            for i, n in enumerate(names):
                s = summarize(*pool([per[i]]), rng, False)
                res["per_period"][n] = {"near_far": s["near_far"], "POD_X": s["POD"]["X"], "n_near": s["NEAR"]["O"]}
    ex = res["variants"]
    ok = all(ex[v]["exploration"]["POD"]["ci_fixed_placebo"][0] > 1 and ex[v]["exploration"]["near_far"]["ci_fixed_placebo"][0] > 1
             for v in ("v_fw", "v_podday", "v_moved"))
    res["R1_reading_exploration"] = "ROBUST" if ok else "NOT_ROBUST"
    json.dump(res, open(out, "w"), indent=1, default=float)
    print(json.dumps({k: res[k] for k in ("R1_reading_exploration", "pod_class_counts")}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
