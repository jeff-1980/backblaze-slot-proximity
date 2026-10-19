"""Daily-location audit (manuscript review item 2, 2026-10-15). Written before running. Descriptive only.

The per-drive tables assign each drive the LAST (vault, pod, slot) seen in the period. This script reads
the daily records and reports how often that assignment can be wrong:
  - drives with more than one distinct (vault, pod, slot) in the period, and how many drive-days are
    spent away from the last location;
  - failed drives whose failure-day location differs from their last location;
  - location-days on which two or more serials report the same (vault, pod, slot) (double occupancy);
  - failed serials that report again after their failure date ("resurrected").
If drives away from their last location account for under 1% of drive-days and of failures, the
last-location simplification is reported as immaterial; otherwise the analysis must be rebuilt on
residency intervals.

USAGE  python dd_residency.py <out.json> <src_dir> <start> <end> [<src_dir> <start> <end> ...]
  src_dir holds daily CSVs (YYYY-MM-DD.csv, recursive) or slim parquet files.
"""
import sys, os, glob, json
import duckdb

con = duckdb.connect()
con.execute("PRAGMA memory_limit='4GB'")
con.execute("PRAGMA threads=2")
con.execute("SET preserve_insertion_order=false")
tmp = os.path.expanduser("~/dd/duckdb_tmp"); os.makedirs(tmp, exist_ok=True)
con.execute(f"PRAGMA temp_directory='{tmp}'")
con.execute("SET max_temp_directory_size='10GB'")


def source(src, start, end):
    files = sorted(f for f in glob.glob(os.path.join(src, "**", "*.csv"), recursive=True)
                   if not os.path.basename(f).startswith("._") and start <= os.path.basename(f)[:10] <= end)
    if files:
        fl = "[" + ", ".join(f"'{f}'" for f in files) + "]"
        raw = f"read_csv({fl}, union_by_name=true, header=true, all_varchar=true, ignore_errors=true)"
    else:
        pq = sorted(glob.glob(os.path.join(src, "*.parquet")))
        fl = "[" + ", ".join(f"'{f}'" for f in pq) + "]"
        raw = f"read_parquet({fl}, union_by_name=true)"
    return f"""SELECT CAST(date AS DATE) AS d, serial_number AS s,
                      TRY_CAST(vault_id AS DOUBLE) AS v, TRY_CAST(pod_id AS DOUBLE) AS p,
                      TRY_CAST(pod_slot_num AS DOUBLE) AS sl, TRY_CAST(failure AS INTEGER) AS f
               FROM {raw}
               WHERE CAST(date AS DATE) BETWEEN DATE '{start}' AND DATE '{end}'
                 AND vault_id IS NOT NULL AND vault_id <> '' AND pod_id IS NOT NULL AND pod_id <> ''
                 AND pod_slot_num IS NOT NULL AND pod_slot_num <> ''"""


def audit(src, start, end):
    con.execute(f"CREATE OR REPLACE TEMP TABLE r AS {source(src, start, end)}")
    q = lambda sql: con.execute(sql).fetchall()
    out = {"src": src, "start": start, "end": end}
    out["drive_days"], out["drives"] = q("SELECT count(*), count(DISTINCT s) FROM r")[0]
    con.execute("""CREATE OR REPLACE TEMP TABLE lastloc AS
                   SELECT s, arg_max(v, d) lv, arg_max(p, d) lp, arg_max(sl, d) lsl, count(DISTINCT (v, p, sl)) nloc,
                          min(CASE WHEN f = 1 THEN d END) fd FROM r GROUP BY s""")
    out["drives_multi_location"] = q("SELECT count(*) FROM lastloc WHERE nloc > 1")[0][0]
    out["drive_days_away_from_last"] = q("""SELECT count(*) FROM r JOIN lastloc USING (s)
                                           WHERE r.v <> lv OR r.p <> lp OR r.sl <> lsl""")[0][0]
    out["failures"] = q("SELECT count(*) FROM lastloc WHERE fd IS NOT NULL")[0][0]
    out["failures_loc_differs"] = q("""SELECT count(*) FROM r JOIN lastloc USING (s)
                                       WHERE r.d = lastloc.fd AND (r.v <> lv OR r.p <> lp OR r.sl <> lsl)""")[0][0]
    out["failures_resurrected"] = q("""SELECT count(DISTINCT s) FROM r JOIN lastloc USING (s)
                                       WHERE lastloc.fd IS NOT NULL AND r.d > lastloc.fd""")[0][0]
    dbl = q("""SELECT count(*), coalesce(sum(n), 0) FROM
               (SELECT d, v, p, sl, count(DISTINCT s) n FROM r GROUP BY ALL HAVING count(DISTINCT s) > 1)""")[0]
    out["double_occupancy_location_days"], out["double_occupancy_drive_days"] = dbl
    out["share_drive_days_away"] = out["drive_days_away_from_last"] / max(out["drive_days"], 1)
    out["share_failures_loc_differs"] = out["failures_loc_differs"] / max(out["failures"], 1)
    con.execute("DROP TABLE r"); con.execute("DROP TABLE lastloc")
    print(json.dumps(out), flush=True)
    return out


if __name__ == "__main__":
    a = sys.argv[2:]
    res = [audit(*a[i:i + 3]) for i in range(0, len(a), 3)]
    imm = all(r["share_drive_days_away"] < 0.01 and r["share_failures_loc_differs"] < 0.01 for r in res)
    json.dump({"periods": res, "reading": "IMMATERIAL" if imm else "REBUILD_ON_RESIDENCY"}, open(sys.argv[1], "w"), indent=1)
    print("reading", "IMMATERIAL" if imm else "REBUILD_ON_RESIDENCY")
