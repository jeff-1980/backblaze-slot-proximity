"""Sample flow by period (Supplementary Table 6), from the derived per-drive tables (k*.parquet).
Added 2026-10-19 (eighth review) to give Supplementary Table 6 its own entry point; it recomputes sample_flow.json.
Per period: drives in the table, drives with a server location, drives with a slot number, servers under the
corrected key (cluster, vault, pod) and the original key (vault, pod), vaults, servers by capacity class (the
smallest of 26, 45 and 60 that is at least the highest slot number seen + 1), index failures (failures of drives with
a slot number) and data centres. The original key and the vault count use drives with a slot number. This reproduces
the archived sample_flow.json exactly.
USAGE  python dd_sample_flow.py <out.json> <k2023h2c.parquet> <k2024.parquet> <k2025.parquet> <k2026c.parquet>
"""
import sys, os, json
import pandas as pd


def flow(path):
    d = pd.read_parquet(path, columns=["vault", "vault_raw", "pod", "slot", "fail", "datacenter"])
    srv = d.dropna(subset=["vault", "pod"])                     # vault = cluster * 100000 + vault_raw
    sl = d.dropna(subset=["vault", "pod", "slot"])
    top = sl.groupby(["vault", "pod"]).slot.max() + 1
    cls = pd.cut(top, [0, 26, 45, 60, 1000], labels=["26", "45", "60", ">60"]).value_counts()
    classes = {k: int(cls.get(k, 0)) for k in ("60", "45", "26", ">60")}
    return {"period": os.path.basename(path).replace(".parquet", ""), "drives": int(len(d)), "drives_with_server": int(len(srv)),
            "drives_with_slot": int(len(sl)), "servers_new_key": int(srv.groupby(["vault", "pod"]).ngroups),
            "servers_with_slot": int(len(top)), "servers_old_key": int(sl.groupby(["vault_raw", "pod"]).ngroups),
            "vaults": int(sl.vault.nunique()), "classes": classes,
            "index_failures": int(sl.fail.sum()), "dcs": sorted(sl.datacenter.dropna().unique().tolist())}


if __name__ == "__main__":
    out = [flow(p) for p in sys.argv[2:]]
    json.dump(out, open(sys.argv[1], "w"), indent=1)
    for o in out:
        print(o)
