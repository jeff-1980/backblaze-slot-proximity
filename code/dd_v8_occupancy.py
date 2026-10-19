"""Occupancy-conflict audit of the reconstructed residences (seventh review, 2026-10-19). Written before running.
Uses dd_v7_cohort.load, so the reconstruction is exactly that of the cohort analysis under the same environment
switches (REPL, GAPI, GAPE). For every slot (cluster, vault, pod, slot) it counts location-days on which two or more
distinct drives are placed by the reconstruction:
  observed     conflicts among observed days only (located daily records);
  all          conflicts after interpolation and extension;
  by type      location-days of the 'all' conflicts that involve at least one interpolated or extended day.
It also checks the index locations: for each index failure, whether the failed drive itself is placed in its slot on
the index day (observed or extended day), and whether another drive is placed in that slot on that day; the latter is
the distance-0 exposure (the current occupant of the historical index slot), and both together are a conflict.
USAGE  REPL=1 python dd_v8_occupancy.py <out.json> <runs.parquet> <k.parquet> [... four pairs]
"""
import sys, os, json, importlib.util
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("v7", os.path.join(HERE, "dd_v7_cohort.py"))
v7 = importlib.util.module_from_spec(spec); spec.loader.exec_module(v7)


def conflict_days(sg):
    """Location-days covered by >= 2 distinct serials; returns (total location-days covered, conflict days)."""
    if len(sg) == 0:
        return 0, 0
    key = pd.factorize(sg.vault.astype(np.int64).astype(str) + "|" + sg.p.astype(str) + "|" + sg.sl.astype(str))[0]
    # merge same-serial pieces at one slot first, so one drive is never counted twice on a day
    d = pd.DataFrame({"k": key, "s": sg.s.values, "t0": sg.t0.values, "t1": sg.t1.values}).sort_values(["k", "s", "t0"])
    k, t, dl = np.concatenate([d.k.values, d.k.values]), np.concatenate([d.t0.values, d.t1.values + 1]), \
        np.concatenate([np.ones(len(d), int), -np.ones(len(d), int)])
    o = np.lexsort((dl, t, k)); k, t, dl = k[o], t[o], dl[o]
    c = np.cumsum(dl)
    span = np.diff(t, append=t[-1]); same = np.append(k[1:] == k[:-1], False)
    span = np.where(same, span, 0)
    return int((span * (c >= 1)).sum()), int((span * (c >= 2)).sum())


def main():
    out = {"config": {"REPL": v7.REPL, "GAPI": v7.GAPI, "GAPE": v7.GAPE}, "periods": []}
    a = sys.argv[2:]
    for runs, kp in zip(a[0::2], a[1::2]):
        sg, li, meta = v7.load(runs, kp)
        cov_o, dup_o = conflict_days(sg[sg.typ == 0])
        cov_a, dup_a = conflict_days(sg)
        # index locations
        key = lambda df: df.vault.astype(np.int64).astype(str) + "|" + df.p.astype(str) + "|" + df.sl.astype(str)
        sgk = sg.assign(key=key(sg)); lik = li.assign(key=key(li))
        m = lik[["s", "key", "t", "ityp"]].reset_index(drop=True).reset_index().merge(sgk[["s", "key", "t0", "t1"]], on="key", suffixes=("", "_o"))
        m = m[(m.t0 <= m.t) & (m.t1 >= m.t)]
        self_ = set(m.loc[m.s_o == m.s, "index"]); other = set(m.loc[m.s_o != m.s, "index"])
        ib = {}
        for nm, v in (("obs", 0), ("ext", 2), ("ext_blocked", 3)):
            ix = set(np.where(lik.ityp.values == v)[0])
            ib[nm] = {"n": len(ix), "failed_drive_placed": len(ix & self_), "other_drive_placed": len(ix & other),
                      "both": len(ix & self_ & other)}
        out["periods"].append({"runs": runs, "k": kp, "location_days_observed": cov_o, "conflict_days_observed": dup_o,
                               "location_days_all": cov_a, "conflict_days_all": dup_a,
                               "conflict_days_added_by_reconstruction": dup_a - dup_o, "index_locations": ib})
        print(runs, cov_o, dup_o, cov_a, dup_a, ib, flush=True)
    json.dump(out, open(sys.argv[1], "w"), indent=1)


if __name__ == "__main__":
    main()
