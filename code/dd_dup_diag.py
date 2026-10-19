"""Follow-up to dd_residency (2026-10-15). Descriptive diagnostics, written before running.
(1) DUP: on sample days, which columns differ between the two serials that share a (vault, pod, slot)?
    If the pairs differ systematically in datacenter / cluster_id / is_legacy_format, (vault_id, pod_id)
    is not a unique server key and the pod definition must add that column.
(2) FAILLOC: failed drives whose failure-day location differs from their last location in the period,
    written as CSV (serial, fdate, failure-day v/p/slot, last v/p/slot) so the per-drive tables can be
    corrected to the failure-day location.
USAGE  python dd_dup_diag.py dup <out.json> <csv_or_parquet_file> [...]
       python dd_dup_diag.py failloc <out.csv> <src_dir> <start> <end>
"""
import sys, os, glob, json
import duckdb

con = duckdb.connect()
con.execute("PRAGMA memory_limit='4GB'"); con.execute("PRAGMA threads=2")
con.execute("SET preserve_insertion_order=false")
tmp = os.path.expanduser("~/dd/duckdb_tmp"); os.makedirs(tmp, exist_ok=True)
con.execute(f"PRAGMA temp_directory='{tmp}'"); con.execute("SET max_temp_directory_size='10GB'")


def rd(f):
    return (f"read_parquet('{f}')" if f.endswith(".parquet")
            else f"read_csv('{f}', header=true, all_varchar=true, ignore_errors=true)")


def dup(out, files):
    res = []
    for f in files:
        con.execute(f"CREATE OR REPLACE TEMP TABLE r AS SELECT * FROM {rd(f)} WHERE pod_slot_num IS NOT NULL AND pod_slot_num <> ''")
        cols = [c[0] for c in con.execute("DESCRIBE r").fetchall()]
        extra = [c for c in ("datacenter", "cluster_id", "is_legacy_format", "model", "capacity_bytes") if c in cols]
        days = [d[0] for d in con.execute("SELECT DISTINCT date FROM r ORDER BY 1 DESC LIMIT 1").fetchall()]
        con.execute(f"CREATE OR REPLACE TEMP TABLE r1 AS SELECT * FROM r WHERE date = '{days[0]}'")
        tot = con.execute("SELECT count(*), count(DISTINCT (vault_id, pod_id, pod_slot_num)) FROM r1").fetchone()
        con.execute("""CREATE OR REPLACE TEMP TABLE dd AS SELECT r1.* FROM r1 JOIN
                       (SELECT vault_id, pod_id, pod_slot_num FROM r1 GROUP BY ALL HAVING count(*) > 1) k USING (vault_id, pod_id, pod_slot_num)""")
        o = {"file": f, "day": str(days[0]), "rows": tot[0], "distinct_locations": tot[1],
             "rows_in_shared_locations": con.execute("SELECT count(*) FROM dd").fetchone()[0], "differs": {}}
        for c in extra:
            o["differs"][c] = con.execute(f"""SELECT count(*) FROM (SELECT vault_id, pod_id, pod_slot_num,
                                              count(DISTINCT {c}) n FROM dd GROUP BY ALL) WHERE n > 1""").fetchone()[0]
        o["n_shared_locations"] = con.execute("SELECT count(*) FROM (SELECT DISTINCT vault_id, pod_id, pod_slot_num FROM dd)").fetchone()[0]
        if "datacenter" in cols:
            o["key_with_datacenter_unique"] = con.execute("""SELECT count(*) = count(DISTINCT (datacenter, vault_id, pod_id, pod_slot_num)) FROM r1""").fetchone()[0]
        sel = ", ".join(["serial_number", "vault_id", "pod_id", "pod_slot_num"] + extra)
        o["examples"] = [list(map(str, x)) for x in con.execute(f"SELECT {sel} FROM dd ORDER BY vault_id, pod_id, pod_slot_num LIMIT 12").fetchall()]
        o["pods_per_vault"] = con.execute("""SELECT n, count(*) FROM (SELECT vault_id, count(DISTINCT pod_id) n FROM r1 GROUP BY 1) GROUP BY 1 ORDER BY 2 DESC LIMIT 6""").fetchall()
        res.append(o); print(json.dumps(o, default=str), flush=True)
    json.dump(res, open(out, "w"), indent=1, default=str)


def failloc(out, src, start, end):
    files = sorted(f for f in glob.glob(os.path.join(src, "**", "*.csv"), recursive=True)
                   if not os.path.basename(f).startswith("._") and start <= os.path.basename(f)[:10] <= end)
    if files:
        raw = "read_csv([" + ", ".join(f"'{f}'" for f in files) + "], union_by_name=true, header=true, all_varchar=true, ignore_errors=true)"
    else:
        raw = "read_parquet([" + ", ".join(f"'{f}'" for f in sorted(glob.glob(os.path.join(src, '*.parquet')))) + "], union_by_name=true)"
    con.execute(f"""CREATE OR REPLACE TEMP TABLE r AS SELECT CAST(date AS DATE) d, serial_number s,
        TRY_CAST(vault_id AS DOUBLE) v, TRY_CAST(pod_id AS DOUBLE) p, TRY_CAST(pod_slot_num AS DOUBLE) sl, TRY_CAST(failure AS INTEGER) f
        FROM {raw} WHERE CAST(date AS DATE) BETWEEN DATE '{start}' AND DATE '{end}'
        AND vault_id IS NOT NULL AND vault_id <> '' AND pod_id IS NOT NULL AND pod_id <> '' AND pod_slot_num IS NOT NULL AND pod_slot_num <> ''""")
    con.execute(f"""COPY (
        WITH fl AS (SELECT s, min(d) fd FROM r WHERE f = 1 GROUP BY s),
             fa AS (SELECT r.s, fl.fd, arg_min(r.v, r.d) fv, arg_min(r.p, r.d) fp, arg_min(r.sl, r.d) fsl
                    FROM r JOIN fl ON r.s = fl.s AND r.d = fl.fd GROUP BY ALL),
             lst AS (SELECT r.s, arg_max(v, d) lv, arg_max(p, d) lp, arg_max(sl, d) lsl FROM r JOIN fl USING (s) GROUP BY r.s)
        SELECT fa.s AS serial_number, fd, fv, fp, fsl, lv, lp, lsl FROM fa JOIN lst USING (s)
        WHERE fv <> lv OR fp <> lp OR fsl <> lsl) TO '{out}' (HEADER)""")
    print(out, con.execute(f"SELECT count(*) FROM read_csv('{out}')").fetchone()[0])


if __name__ == "__main__":
    if sys.argv[1] == "dup":
        dup(sys.argv[2], sys.argv[3:])
    else:
        failloc(*sys.argv[2:6])
