"""Slim one Backblaze quarter zip into one zstd parquet, one daily CSV at a time (2026-10-15).
usage: dd_slim_k.py <zip> <out.parquet> <tmpdir>
Columns: date, serial_number, model, failure, datacenter, cluster_id, vault_id, pod_id, pod_slot_num,
smart_9_raw, all varchar; read_csv options as before (union_by_name, header, all_varchar, ignore_errors).
Each extracted CSV is deleted right after conversion; aborts if free space at $FREE_CHECK_PATH (default: tmpdir) would drop below 10 GB."""
import sys, os, zipfile, shutil, json, time
import duckdb, pyarrow as pa, pyarrow.parquet as pq

COLS = ["date", "serial_number", "model", "failure", "datacenter", "cluster_id", "vault_id", "pod_id",
        "pod_slot_num", "smart_9_raw"]
zp, out, tmpdir = sys.argv[1:4]
os.makedirs(tmpdir, exist_ok=True); os.makedirs(os.path.dirname(out), exist_ok=True)
con = duckdb.connect(); con.execute("PRAGMA memory_limit='3GB'"); con.execute("PRAGMA threads=2")
schema = pa.schema([(c, pa.string()) for c in COLS])
zf = zipfile.ZipFile(zp)
mem = sorted((m for m in zf.infolist() if m.filename.endswith(".csv") and "__MACOSX" not in m.filename
              and not os.path.basename(m.filename).startswith("._")), key=lambda m: os.path.basename(m.filename))
w = pq.ParquetWriter(out + ".part", schema, compression="zstd")
nrows, missing, t0 = 0, {}, time.time()
for i, m in enumerate(mem):
    free = shutil.disk_usage(os.environ.get("FREE_CHECK_PATH", tmpdir)).free
    if free - m.file_size < 10 * 2**30:
        sys.exit(f"ABORT: C: free {free/2**30:.1f} GB, day CSV {m.file_size/2**20:.0f} MB would break 10 GB floor")
    f = os.path.join(tmpdir, os.path.basename(m.filename))
    with zf.open(m) as src, open(f, "wb") as dst:
        shutil.copyfileobj(src, dst, 16 * 2**20)
    rel = f"read_csv(['{f}'], union_by_name=true, header=true, all_varchar=true, ignore_errors=true)"
    have = {r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {rel}").fetchall()}
    sel = ", ".join(c if c in have else f"CAST(NULL AS VARCHAR) AS {c}" for c in COLS)
    for c in COLS:
        if c not in have: missing[c] = missing.get(c, 0) + 1
    t = con.execute(f"SELECT {sel} FROM {rel}").arrow()
    if isinstance(t, pa.RecordBatchReader): t = t.read_all()
    w.write_table(t.cast(schema)); nrows += t.num_rows
    os.remove(f)
    if i % 15 == 0: print(f"  {os.path.basename(m.filename)} rows={t.num_rows} C_free={free/2**30:.1f}GB", flush=True)
w.close(); os.replace(out + ".part", out)
print(json.dumps({"zip": zp, "out": out, "n_days": len(mem), "n_rows": nrows, "days_missing_col": missing,
                  "out_MB": round(os.path.getsize(out) / 2**20, 1), "sec": round(time.time() - t0)}), flush=True)
