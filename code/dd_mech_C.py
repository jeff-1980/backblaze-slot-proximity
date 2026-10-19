"""Mechanism part C (EXPLORATORY), written 2026-10-15 after parts A/B came back unresolved and BEFORE
running C. Question: is the adjacent-slot excess an artefact of technician visits (when the failed
drive is swapped, nearby drives with problems get pulled and logged as failures the same day)?

For each index failure i (day t) whose slot got a replacement on day r (first new drive in the same
vault/pod/slot within 60 d), take follow-on failures of pod-mates j in t+1..t+30 and of REST drives
(same vault, other pod, other slot) in the same window. VISIT-coincident = |fdate_j - r| <= 1.
  C1 share of NEAR (|slot diff| 1-2) follow-on failures that are visit-coincident,
     vs share among FAR (>=16) pod-mates and among REST drives. Vault-clustered bootstrap CI of
     (NEAR share - FAR share).
  C2 NEAR excess recomputed after dropping visit-coincident follow-on failures (and their E for
     days r-1..r+1): X_near_novisit vs X_far_novisit (O/E, same baseline), ratio with bootstrap CI.
Reading fixed now: if C1 difference > 0 (CI excl. 0) AND the C2 ratio CI includes 1, the adjacency
effect is attributed mainly to technician-visit removals (operational artefact). If C2 ratio stays > 1
with CI lower > 1, the effect is not explained by visit-coincident removals.

SMOKE TEST 2026-10-15: independent synthetic failures with next-day replacements -> C1 diff 0.000,
  C2 ratio 1.05 [0.64, 1.58], reading 'ambiguous' (no effect to attribute), as expected.

USAGE  python dd_mech_C.py <out.json> <p2023h2.parquet> <p2024.parquet> <p2025.parquet>
"""
import sys, os, json, importlib.util
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("ddc", os.path.join(HERE, "dd_contagion.py"))
ddc = importlib.util.module_from_spec(spec); spec.loader.exec_module(ddc)
rng = np.random.default_rng(20261015)


def main(out, paths):
    rows = []
    for p in paths:
        d, _ = ddc.prep(p)
        tag = os.path.basename(p)
        for v, ix in d.groupby("vault").indices.items():
            f_, l_, fd_, h_ = d["first"].values[ix], d["last"].values[ix], d.fdate.values[ix], d.h.values[ix]
            p_, s_ = d.pod.values[ix], d.slot.values[ix]
            for j in np.where(~np.isnan(fd_))[0]:
                t = fd_[j]
                rep = (p_ == p_[j]) & (s_ == s_[j]) & (f_ > t) & (f_ <= t + 60)
                if not rep.any():
                    continue
                r = f_[rep].min()
                other = np.ones(len(ix), bool); other[j] = False
                same_pod = p_ == p_[j]; dist = np.abs(s_ - s_[j])
                sets = {"NEAR": other & same_pod & (dist >= 1) & (dist <= 2),
                        "FAR": other & same_pod & (dist >= 16),
                        "REST": other & ~same_pod & (s_ != s_[j])}
                row = {"unit": f"{tag}|{v}"}
                a, b = t + 1, t + 30
                for k, m in sets.items():
                    ev = m & (fd_ >= a) & (fd_ <= b)
                    vis = ev & (np.abs(fd_ - r) <= 1)
                    st = np.maximum(f_[m], a); en = np.minimum(l_[m], b)
                    dd = np.clip(en - st + 1, 0, None).astype(float)
                    # days of the window inside r-1..r+1 (removed in C2)
                    vs = np.maximum(st, r - 1); ve = np.minimum(en, r + 1)
                    vdd = np.clip(ve - vs + 1, 0, None)
                    row[f"n_{k}"] = float(ev.sum()); row[f"v_{k}"] = float(vis.sum())
                    row[f"O_{k}"] = float(ev.sum() - vis.sum())
                    row[f"E_{k}"] = float((h_[m] * (dd - vdd)).sum())
                    row[f"Oall_{k}"] = float(ev.sum()); row[f"Eall_{k}"] = float((h_[m] * dd).sum())
                rows.append(row)
    R = pd.DataFrame(rows).groupby("unit").sum()
    A = R.values; cols = list(R.columns); c = {k: i for i, k in enumerate(cols)}

    def stats(x):
        sh = {k: x[:, c[f"v_{k}"]].sum() / max(x[:, c[f"n_{k}"]].sum(), 1e-12) for k in ("NEAR", "FAR", "REST")}
        X = {k: x[:, c[f"O_{k}"]].sum() / max(x[:, c[f"E_{k}"]].sum(), 1e-12) for k in ("NEAR", "FAR", "REST")}
        Xa = {k: x[:, c[f"Oall_{k}"]].sum() / max(x[:, c[f"Eall_{k}"]].sum(), 1e-12) for k in ("NEAR", "FAR", "REST")}
        return sh, X, Xa
    sh, X, Xa = stats(A)
    bd, br = [], []
    for _ in range(1000):
        s = A[rng.integers(0, len(A), len(A))]
        a, b, _ = stats(s)
        bd.append(a["NEAR"] - a["FAR"]); br.append(b["NEAR"] / max(b["FAR"], 1e-12))
    res = {"n_vault_units": int(len(R)), "n_index_with_rep": int(len(rows)),
           "followon_counts": {k: {"n": float(R[f"n_{k}"].sum()), "visit": float(R[f"v_{k}"].sum())} for k in ("NEAR", "FAR", "REST")},
           "C1_visit_share": sh, "C1_diff_near_far": {"est": sh["NEAR"] - sh["FAR"], "lo": float(np.percentile(bd, 2.5)), "hi": float(np.percentile(bd, 97.5))},
           "OE_all": Xa, "OE_novisit": X,
           "C2_ratio_near_far_novisit": {"est": X["NEAR"] / X["FAR"], "lo": float(np.percentile(br, 2.5)), "hi": float(np.percentile(br, 97.5))}}
    res["reading"] = ("artefact" if (res["C1_diff_near_far"]["lo"] > 0 and res["C2_ratio_near_far_novisit"]["lo"] <= 1)
                      else "not_explained_by_visits" if res["C2_ratio_near_far_novisit"]["lo"] > 1 else "ambiguous")
    json.dump(res, open(out, "w"), indent=2, default=float)
    print(json.dumps(res, indent=2, default=float))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
