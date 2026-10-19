#!/bin/bash
# Sixth-review PC step (2026-10-19). CPU only, no downloads, no deletes, writes only new files in deepdive/v7states/.
# Records without a full location for drives that ever had one (dd_v7_states.py), one period per call.
# Stops if available memory < 6 GB or free space on C: < 20 GB.
cd /mnt/c/Users/ThinkPad/claude_project/deepdive
P=~/bbvenv/bin/python
mkdir -p v7states
ok(){ a=$(free -m | awk '/Mem/{print $7}'); c=$(df -BG --output=avail /mnt/c | tail -1 | tr -dc 0-9); echo "avail_MB=$a C_free_GB=$c"; [ "$a" -ge 6144 ] && [ "$c" -ge 20 ]; }
{
for x in "k2023h2 $HOME/bb2023h2_slim_k 2023-07-01 2023-12-31" "k2024 $HOME/bb2024 2024-01-01 2024-12-31" \
         "k2025 $HOME/bb2025 2025-01-01 2025-12-31" "k2026 $HOME/bb2026_slim_k 2026-01-01 2026-06-30"; do
  set -- $x
  echo "### $(date -Is) START states $1 $(ok)"; ok >/dev/null || { echo "### STOP resources"; exit 1; }
  /usr/bin/time -f "### WALL %e s MAXRSS %M kB EXIT %x" $P -u dd_v7_states.py v7states v6runs "$@"
done
echo "### DONE $(date -Is)"
} >> v7states_master.txt 2>&1
