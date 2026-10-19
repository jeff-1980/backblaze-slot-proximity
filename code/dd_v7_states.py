"""Daily record states for drives that ever had a location (sixth review, 2026-10-19). Written before running. Descriptive.
For every serial in runs_<label>.parquet (dd_v6_runs.py), reads the daily records of the period that carry NO full
location (cluster_id, vault_id, pod_id or pod_slot_num missing or empty) and writes their runs of consecutive days
with failure flags. Together with the located runs this splits each drive's days into four states:
located record, record without location, failure record (with or without location), and no record at all.
Output: <out_dir>/unloc_<label>.parquet with columns s, t0, t1, n_days, n_fail (failure records in the run), and
<out_dir>/unloc_<label>.json with counts (rows, serials, runs, failure records without location).
USAGE  python dd_v7_states.py <out_dir> <runs_dir> <label> <src_dir> <start> <end> [<label> <src_dir> <start> <end> ...]
"""
import sys, os, glob, json
import duckdb

con = duckdb.connect()
con.execute("PRAGMA memory_limit='6GB'"); con.execute("PRAGMA threads=2")
con.execute("SET preserve_insertion_order=false")
tmp = os.path.expanduser("~/dd/duckdb_tmp"); os.makedirs(tmp, exist_ok=True)
con.execute(f"PRAGMA temp_directory='{tmp}'"); con.execute("SET max_temp_directory_size='10GB'")


def raw(src, start, end):
    files = sorted(f for f in glob.glob(os.path.join(src, "**", "*.csv"), recursive=True)
                   if not os.path.basename(f).startswith("._") and start <= os.path.basename(f)[:10] <= end)
    if files:
        return "read_csv([" + ", ".join(f"'{f}'" for f in files) + "], union_by_name=true, header=true, all_varchar=true, ignore_errors=true)"
    return "read_parquet([" + ", ".join(f"'{f}'" for f in sorted(glob.glob(os.path.join(src, '*.parquet')))) + "], union_by_name=true)"


def states(out_dir, runs_dir, label, src, start, end):
    con.execute(f"CREATE OR REPLACE TEMP TABLE ser AS SELECT DISTINCT s FROM read_parquet('{os.path.join(runs_dir, f'runs_{label}.parquet')}')")
    blank = lambda c: f"({c} IS NULL OR CAST({c} AS VARCHAR) = '')"
    con.execute(f"""CREATE OR REPLACE TEMP TABLE u AS
        SELECT DISTINCT CAST(date AS DATE) d, serial_number s, max(TRY_CAST(failure AS INTEGER)) OVER (PARTITION BY serial_number, CAST(date AS DATE)) f
        FROM {raw(src, start, end)}
        WHERE CAST(date AS DATE) BETWEEN DATE '{start}' AND DATE '{end}'
          AND serial_number IN (SELECT s FROM ser)
          AND ({blank('cluster_id')} OR {blank('vault_id')} OR {blank('pod_id')} OR {blank('pod_slot_num')})""")
    con.execute("""CREATE OR REPLACE TEMP TABLE isl AS
        WITH x AS (SELECT *, d - CAST(row_number() OVER (PARTITION BY s ORDER BY d) AS INTEGER) g FROM u)
        SELECT s, min(d) t0, max(d) t1, count(*) n_days, sum(CASE WHEN f = 1 THEN 1 ELSE 0 END) n_fail FROM x GROUP BY s, g""")
    q = lambda sql: con.execute(sql).fetchall()[0]
    o = {"label": label, "src": src, "start": start, "end": end}
    o["rows"], o["serials"] = q("SELECT count(*), count(DISTINCT s) FROM u")
    o["runs"] = q("SELECT count(*) FROM isl")[0]
    o["failure_records_without_location"] = q("SELECT count(*) FROM u WHERE f = 1")[0]
    con.execute(f"COPY isl TO '{os.path.join(out_dir, f'unloc_{label}.parquet')}' (FORMAT parquet)")
    json.dump(o, open(os.path.join(out_dir, f"unloc_{label}.json"), "w"), indent=1, default=str)
    print(json.dumps(o, default=str), flush=True)
    for t in ("ser", "u", "isl"):
        con.execute(f"DROP TABLE IF EXISTS {t}")


if __name__ == "__main__":
    out_dir, runs_dir = sys.argv[1], sys.argv[2]; os.makedirs(out_dir, exist_ok=True); a = sys.argv[3:]
    for i in range(0, len(a), 4):
        states(out_dir, runs_dir, *a[i:i + 4])
