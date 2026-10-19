"""Fourth-review analyses (2026-10-18). Written before reading the outputs.
Part U (from the unified run, dd_unified.py):
  U1 direct comparison of near/far ratios at short (1-30 d) and long (61-180 d; also 31-180 d) separations:
     ratio of ratios with a joint vault bootstrap (same resample for both), two-sided bootstrap p.
  U2 full-uncertainty interval for the near/far ratio and the cluster-0 contrast: in each bootstrap resample
     the placebo mean is re-estimated on the resampled vaults from a resample of the placebo draws. The
     baseline hazard is stratified by vault, so resampling whole vaults leaves every hazard unchanged.
  U3 permutation p with the finite-sample correction (k + 1) / (n + 1).
  U4 one-sided bootstrap p for H1-H7 on 2026H1 ((#draws <= 1) + 1) / (B + 1) and Holm adjustment.
Part V (from dd_v5_riskset.py): near/far ratio and pod X for ALL, RS (index-time risk set), Q1, Q2, and the
  2026H1 table without the flagged models; absolute risk and censoring share under RS.
USAGE  python dd_v5_analyse.py <unified_prefix> <v5_prefix> <out.json>
"""
import sys, json
import numpy as np

B = 1000; SEED = 20261021


def main(up, vp, outp):
    rng = np.random.default_rng(SEED)
    a = np.load(up + ".npz"); m = json.load(open(up + "_meta.json"))
    S, W = m["sets"], m["wins"]; KEEP = [tuple(k) for k in m["pl_keep"]]
    P = len(m["periods"]); si = S.index; wi = W.index
    vid = [np.array(p["vault"]) for p in m["periods"]]; U = np.unique(np.concatenate(vid)); uidx = [np.searchsorted(U, v) for v in vid]
    mult = np.stack([np.bincount(rng.integers(0, len(U), len(U)), minlength=len(U)) for _ in range(B)])
    G = {"explore": [0, 1, 2], "confirm": [3], "pooled": [0, 1, 2, 3]}
    res = {"U1": {}, "U2": {}, "U3": {}, "U4": {}}

    def osum(ps, s, ws, kind=None, boot=False, sel=None):
        """Observed (kind None) O and E summed over windows ws; boot -> B-vectors."""
        o = e = 0
        for p in ps:
            Ov = sum(a[f"O{p}"][si(s), wi(w)] for w in ws); Ev = sum(a[f"E{p}"][si(s), wi(w)] for w in ws)
            if sel is not None:
                Ov = Ov * sel[p]; Ev = Ev * sel[p]
            if boot:
                o = o + mult[:, uidx[p]] @ Ov; e = e + mult[:, uidx[p]] @ Ev
            else:
                o += Ov.sum(); e += Ev.sum()
        return o, e

    def pmean(ps, s, ws):
        po = sum(a[f"pl_main_O{p}"][:, si(s), wi(w)] for p in ps for w in ws)
        pe = sum(a[f"pl_main_E{p}"][:, si(s), wi(w)] for p in ps for w in ws)
        return float((po / pe).mean())

    def nf(ps, ws, boot):
        out = []
        for s in ("d1", "d4"):
            o, e = osum(ps, s, ws, boot=boot)
            out.append(o / np.maximum(e, 1e-12) / pmean(ps, s, ws))
        return out[0] / np.maximum(out[1], 1e-12)

    for g, ps in G.items():
        r = {}
        for nm, ws in (("short_1_30", ["P"]), ("long_61_180", ["G1", "G2"]), ("long_31_180", ["L5", "G1", "G2"])):
            r[nm] = {"ratio": float(nf(ps, ws, False)), "boot": nf(ps, ws, True)}
        for ln in ("long_61_180", "long_31_180"):
            rr = r["short_1_30"]["boot"] / np.maximum(r[ln]["boot"], 1e-12)
            pt = r["short_1_30"]["ratio"] / r[ln]["ratio"]
            p2 = 2 * min((rr <= 1).mean(), (rr >= 1).mean()); p2 = max(p2, 2 / (B + 1))
            r[f"short_over_{ln}"] = {"ratio": pt, "lo": float(np.percentile(rr, 2.5)), "hi": float(np.percentile(rr, 97.5)), "p_two_sided": float(min(p2, 1))}
        for nm in ("short_1_30", "long_61_180", "long_31_180"):
            bb = r[nm].pop("boot"); r[nm].update(lo=float(np.percentile(bb, 2.5)), hi=float(np.percentile(bb, 97.5)))
        res["U1"][g] = r

    # U2 full uncertainty using per-vault placebo arrays (keep: d1, d4 at P)
    kd1, kd4 = KEEP.index(("d1", "P")), KEEP.index(("d4", "P"))
    npl = a["pl_main_keep0"].shape[0]

    def full(ps, sel=None):
        draws = rng.integers(0, npl, size=(B, npl))
        xs = {}
        for s, kk in (("d1", kd1), ("d4", kd4)):
            o, e = osum(ps, s, ["P"], boot=True, sel=sel)
            po = np.zeros((B,)); pe_ = np.zeros((B,))
            # placebo ratio per draw on the resampled vaults, then mean over resampled draws
            PO = sum((mult[:, uidx[p]] * (sel[p] if sel is not None else 1)) @ a[f"pl_main_keep{p}"][:, kk, 0, :].T for p in ps)  # B x npl
            PE = sum((mult[:, uidx[p]] * (sel[p] if sel is not None else 1)) @ a[f"pl_main_keep{p}"][:, kk, 1, :].T for p in ps)
            ratio = PO / np.maximum(PE, 1e-12)
            pm = np.take_along_axis(ratio, draws, 1).mean(1)
            xs[s] = o / np.maximum(e, 1e-12) / pm
        return xs["d1"] / np.maximum(xs["d4"], 1e-12)

    for g, ps in (("explore", [0, 1, 2]), ("confirm", [3])):
        rb = full(ps)
        res["U2"][g] = {"lo": float(np.percentile(rb, 2.5)), "hi": float(np.percentile(rb, 97.5))}
    cl = [np.array(m["periods"][p]["cluster"]) == 0 for p in range(P)]
    r0 = full([0, 1, 2, 3], cl); r1 = full([0, 1, 2, 3], [~c for c in cl])
    d = np.log(r0) - np.log(np.maximum(r1, 1e-12))
    res["U2"]["cluster0"] = {"lo": float(np.percentile(r0, 2.5)), "hi": float(np.percentile(r0, 97.5))}
    res["U2"]["other"] = {"lo": float(np.percentile(r1, 2.5)), "hi": float(np.percentile(r1, 97.5))}
    res["U2"]["c0_over_other"] = {"lo": float(np.exp(np.percentile(d, 2.5))), "hi": float(np.exp(np.percentile(d, 97.5))),
                                  "p_two_sided": float(max(2 * min((d <= 0).mean(), (d >= 0).mean()), 2 / (B + 1)))}

    # U3 permutation p corrected
    for g, ps in (("explore", [0, 1, 2]), ("confirm", [3])):
        hr = lambda s: osum(ps, s, ["P"])[0] / osum(ps, s, ["P"])[1]
        obs = hr("d1") / hr("d4"); res["U3"][g] = {}
        for kind in ("C1", "C3"):
            po = {s: sum(a[f"pl_{kind}_O{p}"][:, si(s), wi("P")] for p in ps) for s in ("d1", "d4")}
            pe = {s: sum(a[f"pl_{kind}_E{p}"][:, si(s), wi("P")] for p in ps) for s in ("d1", "d4")}
            pr = (po["d1"] / pe["d1"]) / (po["d4"] / pe["d4"]); k = int((pr >= obs).sum())
            res["U3"][g][kind] = {"n": int(len(pr)), "k_ge_obs": k, "p": (k + 1) / (len(pr) + 1)}

    # U4 bootstrap p for H1-H7 on 2026H1, Holm
    ps = [3]
    def xb(s, ws):
        o, e = osum(ps, s, ws, boot=True); return o / np.maximum(e, 1e-12) / pmean(ps, s, ws)
    stats = {"H1": xb("POD", ["P"]), "H2": xb("TOME", ["P"]), "H3": xb("TOME", ["P"]) / xb("REST", ["P"]),
             "H4": xb("d1", ["P"]) / xb("d4", ["P"]), "H5": xb("TOME", ["E7"]) / xb("TOME", ["LATE"]),
             "H6": xb("TOME", ["M"]), "H7": xb("d1_sb", ["P"]) / xb("d4_sb", ["P"])}
    pv = {k: (int((v <= 1).sum()) + 1) / (B + 1) for k, v in stats.items()}
    order = sorted(pv, key=pv.get); holm = {}; run = 0
    for i, k in enumerate(order):
        run = max(run, min(1, (len(order) - i) * pv[k])); holm[k] = run
    res["U4"] = {"p_one_sided": pv, "holm": holm, "B": B}

    # Part V
    b = np.load(vp + ".npz"); vm = json.load(open(vp + "_meta.json"))
    VS, VW, VV = vm["sets"], vm["wins"], vm["var"]
    vvid = [np.array(p["vault"]) for p in vm["periods"]]
    VU = np.unique(np.concatenate(vvid)); vu = [np.searchsorted(VU, v) for v in vvid]
    vmult = np.stack([np.bincount(rng.integers(0, len(VU), len(VU)), minlength=len(VU)) for _ in range(B)])
    nP = len(vm["periods"])
    VG = {"explore": [0, 1, 2], "confirm": [3], "confirm_xanom": [4]}
    if nP == 2:  # test run on 2026H1 only
        VG = {"confirm": [0], "confirm_xanom": [1]}
    res["V"] = {"flagged_models": vm["flagged_models"], "periods": vm["periods"] and [{k: p[k] for k in p if k != "vault"} for p in vm["periods"]]}

    def vX(ps, vi, s, w="P"):
        o = sum(b[f"O{p}"][vi, VS.index(s), VW.index(w)].sum() for p in ps); e = sum(b[f"E{p}"][vi, VS.index(s), VW.index(w)].sum() for p in ps)
        po = sum(b[f"pl_main_O{p}"][:, vi, VS.index(s), VW.index(w)] for p in ps); pe = sum(b[f"pl_main_E{p}"][:, vi, VS.index(s), VW.index(w)] for p in ps)
        pm = (po / pe).mean()
        ob = sum(vmult[:, vu[p]] @ b[f"O{p}"][vi, VS.index(s), VW.index(w)] for p in ps)
        eb = sum(vmult[:, vu[p]] @ b[f"E{p}"][vi, VS.index(s), VW.index(w)] for p in ps)
        return o / e / pm, ob / np.maximum(eb, 1e-12) / pm, int(o)

    for g, ps in VG.items():
        res["V"][g] = {}
        for vi, vn in enumerate(VV):
            x1, b1, n1 = vX(ps, vi, "d1"); x4, b4, n4 = vX(ps, vi, "d4"); xp, bp, npd = vX(ps, vi, "POD")
            rr = b1 / np.maximum(b4, 1e-12)
            ent = {"near_far": float(x1 / x4), "lo": float(np.percentile(rr, 2.5)), "hi": float(np.percentile(rr, 97.5)),
                   "n_d1": n1, "n_d4": n4, "pod_X": float(xp), "pod_lo": float(np.percentile(bp, 2.5)), "pod_hi": float(np.percentile(bp, 97.5))}
            # C3 within-server
            hr = lambda s: sum(b[f"O{p}"][vi, VS.index(s), 0].sum() for p in ps) / sum(b[f"E{p}"][vi, VS.index(s), 0].sum() for p in ps)
            obs = hr("d1") / hr("d4")
            po1 = sum(b[f"pl_C3_O{p}"][:, vi, VS.index("d1"), 0] for p in ps); pe1 = sum(b[f"pl_C3_E{p}"][:, vi, VS.index("d1"), 0] for p in ps)
            po4 = sum(b[f"pl_C3_O{p}"][:, vi, VS.index("d4"), 0] for p in ps); pe4 = sum(b[f"pl_C3_E{p}"][:, vi, VS.index("d4"), 0] for p in ps)
            pr = (po1 / pe1) / (po4 / pe4)
            ent["C3_normalised"] = float(obs / pr.mean()); ent["C3_p"] = (int((pr >= obs).sum()) + 1) / (len(pr) + 1)
            for kk in ("d1", "d4"):
                k = VS.index(kk)
                ex = sum(b[f"Aexp{p}"][vi, k].sum() for p in ps); ev = sum(b[f"Aev{p}"][vi, k].sum() for p in ps); ce = sum(b[f"Acens{p}"][vi, k].sum() for p in ps)
                ent[f"abs_{kk}"] = {"exposures": int(ex), "failures": int(ev), "per_10k": 1e4 * ev / max(ex, 1), "censored_share": ce / max(ex, 1)}
            res["V"][g][vn] = ent
    json.dump(res, open(outp, "w"), indent=1, default=float)
    print(json.dumps({k: res[k] for k in ("U1", "U2", "U3", "U4")}, indent=1, default=float)[:4000])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
