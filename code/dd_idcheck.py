"""Daily server-identity checks (second review item 1, 2026-10-17). Written before running. Descriptive.
Reads daily records (CSV dirs or slim parquet with cluster_id and datacenter) and reports, per period:
  - days on which a (cluster_id, vault_id) or a (cluster_id, vault_id, pod_id) appears in more than one
    datacenter, and how many such keys;
  - share of (cluster_id, vault_id, pod_id, pod_slot_num) location-days held by two or more serials, with and
    without datacenter in the key;
  - serials whose (cluster_id, vault_id, pod_id) changes between consecutive observed days, how many of those
    changes also change datacenter or cluster only (cluster relabelling), and changes that revert within 7 days;
  - for each serial, the number of distinct cluster_id values (relabelling of cluster without moving).
  - the same counts restricted to 2023 Q3 (Backblaze notes that cluster_id was not always correct then).
USAGE  python dd_idcheck.py <out.json> <src_dir> <start> <end> [<src_dir> <start> <end> ...]
"""
import sys, os, glob, json
import duckdb

con = duckdb.connect()
con.execute("PRAGMA memory_limit='6GB'"); con.execute("PRAGMA threads=2")
con.execute("SET preserve_insertion_order=false")
tmp = os.path.expanduser("~/dd/duckdb_tmp"); os.makedirs(tmp, exist_ok=True)
con.execute(f"PRAGMA temp_directory='{tmp}'"); con.execute("SET max_temp_directory_size='12GB'")


def source(src, start, end):
    files = sorted(f for f in glob.glob(os.path.join(src, "**", "*.csv"), recursive=True)
                   if not os.path.basename(f).startswith("._") and start <= os.path.basename(f)[:10] <= end)
    if files:
        raw = "read_csv([" + ", ".join(f"'{f}'" for f in files) + "], union_by_name=true, header=true, all_varchar=true, ignore_errors=true)"
    else:
        raw = "read_parquet([" + ", ".join(f"'{f}'" for f in sorted(glob.glob(os.path.join(src, '*.parquet')))) + "], union_by_name=true)"
    return f"""SELECT CAST(date AS DATE) d, serial_number s, datacenter dc, TRY_CAST(cluster_id AS DOUBLE) c,
                      TRY_CAST(vault_id AS DOUBLE) v, TRY_CAST(pod_id AS DOUBLE) p, TRY_CAST(pod_slot_num AS DOUBLE) sl
               FROM {raw} WHERE CAST(date AS DATE) BETWEEN DATE '{start}' AND DATE '{end}'
                 AND vault_id IS NOT NULL AND vault_id <> '' AND pod_id IS NOT NULL AND pod_id <> ''"""


def check(src, start, end):
    con.execute(f"CREATE OR REPLACE TEMP TABLE r AS {source(src, start, end)}")
    q = lambda sql: con.execute(sql).fetchall()
    o = {"src": src, "start": start, "end": end}
    o["rows"], o["serials"], o["days"] = q("SELECT count(*), count(DISTINCT s), count(DISTINCT d) FROM r")[0]
    o["rows_cluster_null"] = q("SELECT count(*) FROM r WHERE c IS NULL")[0][0]
    o["rows_dc_null"] = q("SELECT count(*) FROM r WHERE dc IS NULL OR dc = ''")[0][0]
    for nm, key in (("vault", "c, v"), ("pod", "c, v, p")):
        a = q(f"""SELECT count(*), count(DISTINCT d) FROM (SELECT d, {key} FROM r GROUP BY ALL HAVING count(DISTINCT dc) > 1)""")[0]
        o[f"{nm}_key_days_multi_dc"], o[f"days_with_{nm}_multi_dc"] = a
    for nm, key in (("loc", "c, v, p, sl"), ("loc_dc", "dc, c, v, p, sl")):
        a = q(f"""SELECT count(*), coalesce(sum(CASE WHEN n > 1 THEN 1 ELSE 0 END), 0) FROM
                  (SELECT d, {key}, count(DISTINCT s) n FROM r WHERE sl IS NOT NULL GROUP BY ALL)""")[0]
        o[f"{nm}_days"], o[f"{nm}_days_2plus"] = a
        o[f"{nm}_share_2plus"] = a[1] / max(a[0], 1)
    # 2026-10-18 memory fix (2024 run hit the 12 GB spill cap): the lag window runs only on serials that ever
    # show more than one (dc, cluster, vault, pod); all other serials cannot contribute a change. Same counts.
    con.execute("""CREATE OR REPLACE TEMP TABLE multi AS SELECT s FROM r GROUP BY s
                   HAVING count(DISTINCT (coalesce(dc, ''), c, v, p)) > 1""")
    con.execute("""CREATE OR REPLACE TEMP TABLE ch AS SELECT s, d, dc, c, v, p,
                   lag(d) OVER w pd, lag(dc) OVER w pdc, lag(c) OVER w pc, lag(v) OVER w pv, lag(p) OVER w pp
                   FROM r SEMI JOIN multi USING (s) WINDOW w AS (PARTITION BY s ORDER BY d)""")
    o["serials_multi_key"] = q("SELECT count(*) FROM multi")[0][0]
    o["changes_pod"] = q("SELECT count(*) FROM ch WHERE pd IS NOT NULL AND (c <> pc OR v <> pv OR p <> pp)")[0][0]
    o["changes_cluster_only"] = q("SELECT count(*) FROM ch WHERE pd IS NOT NULL AND c <> pc AND v = pv AND p = pp AND dc = pdc")[0][0]
    o["changes_with_dc_change"] = q("SELECT count(*) FROM ch WHERE pd IS NOT NULL AND dc <> pdc")[0][0]
    o["serials_with_pod_change"] = q("SELECT count(DISTINCT s) FROM ch WHERE pd IS NOT NULL AND (c <> pc OR v <> pv OR p <> pp)")[0][0]
    o["serials_multi_cluster"] = q("SELECT count(*) FROM (SELECT s FROM r GROUP BY s HAVING count(DISTINCT c) > 1)")[0][0]
    o["serials_multi_dc"] = q("SELECT count(*) FROM (SELECT s FROM r GROUP BY s HAVING count(DISTINCT dc) > 1)")[0][0]
    o["reverting_pod_changes_7d"] = q("""SELECT count(*) FROM ch WHERE pd IS NOT NULL AND (c <> pc OR v <> pv OR p <> pp)
        AND EXISTS (SELECT 1 FROM ch r2 WHERE r2.s = ch.s AND r2.c = ch.pc AND r2.v = ch.pv AND r2.p = ch.pp
                    AND r2.d > ch.d AND r2.d <= ch.d + 7)""")[0][0]
    con.execute("DROP TABLE r"); con.execute("DROP TABLE ch"); con.execute("DROP TABLE multi")
    print(json.dumps(o), flush=True)
    return o


if __name__ == "__main__":
    a = sys.argv[2:]
    res = []
    for i in range(0, len(a), 3):
        res.append(check(*a[i:i + 3]))
        if a[i + 1] <= "2023-09-30" and a[i + 2] >= "2023-07-01":
            res.append(check(a[i], "2023-07-01", "2023-09-30"))
    json.dump(res, open(sys.argv[1], "w"), indent=1)
