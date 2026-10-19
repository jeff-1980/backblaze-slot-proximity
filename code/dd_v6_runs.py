"""Daily location runs for the index-day cohort (fifth review, 2026-10-18). Written before running. Descriptive.
Reads the daily records (CSV dirs or slim parquet) and writes, per serial, its location runs: maximal spells of
consecutive daily records with the same (cluster_id, vault_id, pod_id, pod_slot_num). A run ends when the
location changes, when a day is missing, or at the last record. failed = 1 if the run's last record has
failure = 1. Only serials with more than one location or with missing days need the gaps-and-islands step;
all others form one run straight from a GROUP BY (same memory pattern as dd_idcheck.py).
Amendment 2026-10-18 (after the first PC run): 2024-2026 hit the DuckDB spill cap in the first GROUP BY, so
serials are processed in NB hash buckets (each bucket re-reads the source); the parts are kept in parts/ and concatenated;
the spill cap is 10 GB. Results are identical to a single pass because every step is within serial.
Output: <out_dir>/runs_<label>.parquet with columns
  s, c, v, p, sl, dc, model, t0 (first day), t1 (last day), n_days, failed
and <out_dir>/runs_<label>.json with counts.
USAGE  python dd_v6_runs.py <out_dir> <label> <src_dir> <start> <end> [<label> <src_dir> <start> <end> ...]
"""
import sys, os, glob, json
import duckdb

con = duckdb.connect()
con.execute("PRAGMA memory_limit='6GB'"); con.execute("PRAGMA threads=2")
con.execute("SET preserve_insertion_order=false")
tmp = os.path.expanduser("~/dd/duckdb_tmp"); os.makedirs(tmp, exist_ok=True)
con.execute(f"PRAGMA temp_directory='{tmp}'"); con.execute("SET max_temp_directory_size='10GB'")


NB = int(os.environ.get("NB", 8))


def source(src, start, end, k=None):
    files = sorted(f for f in glob.glob(os.path.join(src, "**", "*.csv"), recursive=True)
                   if not os.path.basename(f).startswith("._") and start <= os.path.basename(f)[:10] <= end)
    if files:
        raw = "read_csv([" + ", ".join(f"'{f}'" for f in files) + "], union_by_name=true, header=true, all_varchar=true, ignore_errors=true)"
    else:
        raw = "read_parquet([" + ", ".join(f"'{f}'" for f in sorted(glob.glob(os.path.join(src, '*.parquet')))) + "], union_by_name=true)"
    return f"""SELECT CAST(date AS DATE) d, serial_number s, model, datacenter dc, TRY_CAST(cluster_id AS DOUBLE) c, TRY_CAST(vault_id AS DOUBLE) v,
                      TRY_CAST(pod_id AS DOUBLE) p, TRY_CAST(pod_slot_num AS DOUBLE) sl,
                      TRY_CAST(failure AS INTEGER) f
               FROM {raw} WHERE CAST(date AS DATE) BETWEEN DATE '{start}' AND DATE '{end}'
                 AND vault_id IS NOT NULL AND vault_id <> '' AND pod_id IS NOT NULL AND pod_id <> ''
                 AND pod_slot_num IS NOT NULL AND pod_slot_num <> ''""" + (
               "" if k is None else f" AND hash(serial_number) % {NB} = {k}")


def runs(out_dir, label, src, start, end):
    keys = ("rows", "serials", "serials_irregular", "serials_multi_location", "runs", "failure_records",
            "failed_runs", "slot_changes_within_pod")
    o = {"label": label, "src": src, "start": start, "end": end, "buckets": NB, **{k: 0 for k in keys}}
    parts = []
    for k in range(NB):
        os.makedirs(os.path.join(out_dir, "parts"), exist_ok=True)
        part = os.path.join(out_dir, "parts", f"runs_{label}_part{k}.parquet")
        ob = bucket(src, start, end, k, part)
        for x in keys:
            o[x] += ob[x]
        parts.append(part); print(label, "bucket", k, ob, flush=True)
    path = os.path.join(out_dir, f"runs_{label}.parquet")
    con.execute(f"COPY (SELECT * FROM read_parquet({parts})) TO '{path}' (FORMAT parquet)")
    json.dump(o, open(os.path.join(out_dir, f"runs_{label}.json"), "w"), indent=1, default=str)
    print(json.dumps(o, default=str), flush=True)


def bucket(src, start, end, k, path):
    q = lambda sql: con.execute(sql).fetchall()
    con.execute(f"CREATE OR REPLACE TEMP TABLE r AS {source(src, start, end, k)}")
    # one row per serial and location
    con.execute("""CREATE OR REPLACE TEMP TABLE g AS SELECT s, c, v, p, sl, min(d) t0, max(d) t1,
                   count(DISTINCT d) n_days, any_value(dc) dc, any_value(model) model FROM r GROUP BY s, c, v, p, sl""")
    con.execute("""CREATE OR REPLACE TEMP TABLE irr AS SELECT s FROM g GROUP BY s
                   HAVING count(*) > 1 OR sum(n_days) < max(datediff('day', t0, t1) + 1)""")
    o = {}
    o["rows"], o["serials"] = q("SELECT count(*), count(DISTINCT s) FROM r")[0]
    o["serials_irregular"] = q("SELECT count(*) FROM irr")[0][0]
    o["serials_multi_location"] = q("SELECT count(*) FROM (SELECT s FROM g GROUP BY s HAVING count(*) > 1)")[0][0]
    # regular serials: one run = the group row
    con.execute("""CREATE OR REPLACE TEMP TABLE reg AS SELECT g.* FROM g ANTI JOIN irr USING (s)""")
    # irregular serials: gaps and islands on daily rows
    con.execute("""CREATE OR REPLACE TEMP TABLE isl AS
        WITH x AS (SELECT r.*, lag(d) OVER w pd, lag(c) OVER w pc, lag(v) OVER w pv, lag(p) OVER w pp, lag(sl) OVER w psl
                   FROM r SEMI JOIN irr USING (s) WINDOW w AS (PARTITION BY s ORDER BY d)),
             y AS (SELECT *, sum(CASE WHEN pd IS NULL OR datediff('day', pd, d) > 1 OR c IS DISTINCT FROM pc
                                      OR v <> pv OR p <> pp OR sl <> psl THEN 1 ELSE 0 END)
                             OVER (PARTITION BY s ORDER BY d ROWS UNBOUNDED PRECEDING) k FROM x)
        SELECT s, c, v, p, sl, min(d) t0, max(d) t1, count(DISTINCT d) n_days, any_value(dc) dc,
               any_value(model) model FROM y GROUP BY s, k, c, v, p, sl""")
    con.execute("""CREATE OR REPLACE TEMP TABLE allr AS SELECT * FROM reg UNION ALL SELECT * FROM isl""")
    # failed if the run's last day carries failure = 1
    con.execute("""CREATE OR REPLACE TEMP TABLE fl AS SELECT s, d FROM r WHERE f = 1""")
    con.execute("""CREATE OR REPLACE TEMP TABLE outt AS SELECT a.*, CASE WHEN fl.s IS NULL THEN 0 ELSE 1 END failed
                   FROM allr a LEFT JOIN fl ON a.s = fl.s AND a.t1 = fl.d""")
    o["runs"] = q("SELECT count(*) FROM outt")[0][0]
    o["failure_records"] = q("SELECT count(*) FROM fl")[0][0]
    o["failed_runs"] = q("SELECT sum(failed) FROM outt")[0][0]
    o["slot_changes_within_pod"] = q("""SELECT count(*) FROM (SELECT s, c, v, p FROM g GROUP BY s, c, v, p
                                        HAVING count(DISTINCT sl) > 1)""")[0][0]
    con.execute(f"COPY outt TO '{path}' (FORMAT parquet)")
    for t in ("r", "g", "irr", "reg", "isl", "allr", "fl", "outt"):
        con.execute(f"DROP TABLE IF EXISTS {t}")
    return {x: int(v or 0) for x, v in o.items()}


if __name__ == "__main__":
    out_dir = sys.argv[1]; os.makedirs(out_dir, exist_ok=True); a = sys.argv[2:]
    for i in range(0, len(a), 4):
        runs(out_dir, *a[i:i + 4])
