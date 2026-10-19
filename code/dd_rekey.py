"""Server-key correction (2026-10-15). Written before running.

Finding (dd_dup_diag): vault_id is reused across clusters/datacenters, so (vault_id, pod_id) is not a
unique server and (vault_id, pod_id, pod_slot_num) holds two drives in 13-25% of locations. The cluster_id
column separates every such pair in 2024 and 2025. This script rebuilds the per-drive period tables with
the cluster in the key. Output has the same columns as dd_contagion.build plus cluster, datacenter;
`vault` becomes cluster * 100000 + vault_id, so every downstream script (prep, tally, placebo strata,
bootstrap units) uses the corrected key without code changes.

Modes
  csv  <src_dir> <start> <end> <out.parquet>
       From daily CSVs that carry cluster_id (2024, 2025). Same aggregation as build(); location is the
       last non-null value in the period, as before.
  map  <old_per_drive.parquet> <ref_src_dir> <ref_start> <ref_end> <out.parquet>
       For periods whose slim tables lack cluster_id (2023H2, 2026H1): cluster is taken from the drive's
       serial in a reference window of CSVs (2024-01 for 2023H2, 2025-12 for 2026H1). Drives not found
       get the cluster of their (vault_id, pod_id) if that pair maps to a single cluster in the reference
       window; otherwise they are dropped (reported as n_dropped).
"""
import sys, os, glob, json
import duckdb
import pandas as pd

con = duckdb.connect()
con.execute("PRAGMA memory_limit='6GB'"); con.execute("PRAGMA threads=2")
con.execute("SET preserve_insertion_order=false")
tmp = os.path.expanduser("~/dd/duckdb_tmp"); os.makedirs(tmp, exist_ok=True)
con.execute(f"PRAGMA temp_directory='{tmp}'"); con.execute("SET max_temp_directory_size='12GB'")


def csv_src(src, start, end):
    files = sorted(f for f in glob.glob(os.path.join(src, "**", "*.csv"), recursive=True)
                   if not os.path.basename(f).startswith("._") and start <= os.path.basename(f)[:10] <= end)
    if not files:  # I/O fix 2026-10-15: slim per-quarter parquet input (see dd_fixes.txt)
        pq = sorted(glob.glob(os.path.join(src, "*.parquet")))
        assert pq, "no csv or parquet files in range"
        raw = ("(SELECT * FROM read_parquet([" + ", ".join(f"'{f}'" for f in pq) +
               f"], union_by_name=true) WHERE date >= '{start}' AND date <= '{end}')")
        dates = [r[0] for r in con.execute(f"SELECT DISTINCT date FROM {raw} ORDER BY 1").fetchall()]
        assert dates, "no rows in range"
        return dates, raw
    return files, ("read_csv([" + ", ".join(f"'{f}'" for f in files) +
                   "], union_by_name=true, header=true, all_varchar=true, ignore_errors=true)")


def build_csv(src, start, end, out):
    files, raw = csv_src(src, start, end)
    d = con.execute(f"""
    WITH r AS (SELECT * FROM {raw})
    SELECT serial_number,
           arg_max(model, date) AS model,
           arg_max(TRY_CAST(cluster_id AS DOUBLE), date) AS cluster,
           arg_max(datacenter, date) AS datacenter,
           arg_max(TRY_CAST(vault_id AS DOUBLE), date) AS vault_raw,
           arg_max(TRY_CAST(pod_id AS DOUBLE), date) AS pod,
           arg_max(TRY_CAST(pod_slot_num AS DOUBLE), date) AS slot,
           count(DISTINCT (cluster_id, vault_id, pod_id)) AS n_pods_seen,
           min(CAST(date AS DATE)) AS first, max(CAST(date AS DATE)) AS last,
           max(TRY_CAST(failure AS INTEGER)) AS fail,
           min(CASE WHEN TRY_CAST(failure AS INTEGER)=1 THEN CAST(date AS DATE) END) AS fdate,
           arg_min(TRY_CAST(smart_9_raw AS DOUBLE), date) AS poh0
    FROM r WHERE vault_id IS NOT NULL AND vault_id <> '' AND pod_id IS NOT NULL AND pod_id <> ''
    GROUP BY serial_number""").df()
    finish(d, start, end, len(files), out, {"n_cluster_missing": int(d.cluster.isna().sum())})


def finish(d, start, end, nfiles, out, extra):
    miss = d.cluster.isna()
    d = d[~miss].copy()
    d["vault"] = d.cluster * 100000 + d.vault_raw
    d["period_start"], d["period_end"], d["n_files"] = start, end, nfiles
    d.to_parquet(out, index=False)
    loc = d.dropna(subset=["slot"]).groupby(["vault", "pod", "slot"]).size()
    print(json.dumps({"out": out, "n_drives": len(d), "n_fail": int(d.fail.fillna(0).sum()),
                      "n_vault_units": int(d.vault.nunique()), "n_pods": int(d.groupby(["vault", "pod"]).ngroups),
                      "locations_with_2plus_drives": int((loc > 1).sum()), "dropped_no_cluster": int(miss.sum()), **extra}))


def build_map(old, ref_src, ref_start, ref_end, out):
    _, raw = csv_src(ref_src, ref_start, ref_end)
    ref = con.execute(f"""SELECT serial_number, arg_max(TRY_CAST(cluster_id AS DOUBLE), date) AS c,
                                 arg_max(TRY_CAST(vault_id AS DOUBLE), date) AS v, arg_max(TRY_CAST(pod_id AS DOUBLE), date) AS p
                          FROM {raw} WHERE vault_id IS NOT NULL AND vault_id <> '' GROUP BY serial_number""").df()
    d = pd.read_parquet(old).rename(columns={"vault": "vault_raw"})
    d = d.merge(ref[["serial_number", "c"]].rename(columns={"c": "cluster"}), on="serial_number", how="left")
    by_serial = int(d.cluster.notna().sum())
    vp = ref.dropna(subset=["c"]).groupby(["v", "p"]).c.agg(["nunique", "first"]).reset_index()
    vp = vp[vp["nunique"] == 1][["v", "p", "first"]].rename(columns={"v": "vault_raw", "p": "pod", "first": "c_vp"})
    d = d.merge(vp, on=["vault_raw", "pod"], how="left")
    d["cluster"] = d.cluster.fillna(d.c_vp)
    d = d.drop(columns="c_vp")
    d["datacenter"] = None
    finish(d, d.period_start.iloc[0], d.period_end.iloc[0], int(d.n_files.iloc[0]), out,
           {"assigned_by_serial": by_serial, "n_input": int(len(d)), "n_fail_input": int(d.fail.fillna(0).sum())})


if __name__ == "__main__":
    if sys.argv[1] == "csv":
        build_csv(*sys.argv[2:6])
    else:
        build_map(*sys.argv[2:7])
