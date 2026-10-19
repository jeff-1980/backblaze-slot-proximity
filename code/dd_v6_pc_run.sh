#!/bin/bash
# Fifth-review PC step (2026-10-18). CPU only, no downloads, no deletes, writes only new files in deepdive/v6runs/.
# Builds daily location runs (dd_v6_runs.py), one period per call. Rerun 2026-10-18 for 2024-2026 only (2023H2 done),
# with the bucketed script. Stops if available
# memory < 6 GB or free space on C: < 20 GB (DuckDB may spill up to 10 GB into ~/dd/duckdb_tmp).
cd /mnt/c/Users/ThinkPad/claude_project/deepdive
P=~/bbvenv/bin/python
mkdir -p v6runs
ok(){ a=$(free -m | awk '/Mem/{print $7}'); c=$(df -BG --output=avail /mnt/c | tail -1 | tr -dc 0-9); echo "avail_MB=$a C_free_GB=$c"; [ "$a" -ge 6144 ] && [ "$c" -ge 20 ]; }
{
for x in "k2024 $HOME/bb2024 2024-01-01 2024-12-31" \
         "k2025 $HOME/bb2025 2025-01-01 2025-12-31" "k2026 $HOME/bb2026_slim_k 2026-01-01 2026-06-30"; do
  set -- $x
  echo "### $(date -Is) START runs $1 $(ok)"; ok >/dev/null || { echo "### STOP resources"; exit 1; }
  /usr/bin/time -f "### WALL %e s MAXRSS %M kB EXIT %x" $P -u dd_v6_runs.py v6runs "$@"
done
echo "### DONE $(date -Is)"
} >> v6runs_master.txt 2>&1
