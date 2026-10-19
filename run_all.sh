#!/bin/bash
# Rerun steps 1-5 from the derived tables in data/. Output goes to results_rerun/.
set -euo pipefail
cd "$(dirname "$0")"; P=${PYTHON:-python3}; D=data; O=results_rerun; mkdir -p $O
$P code/dd_overlap_check.py $D/k2023h2c.parquet $D/k2024.parquet $D/k2025.parquet $D/k2026c.parquet
$P code/dd_contagion.py analyze $O/dd_explore_rekeyc.json $D/k2023h2c.parquet $D/k2024.parquet $D/k2025.parquet
$P code/dd_contagion.py analyze $O/dd_confirm_rekeyc.json $D/k2026c.parquet
$P code/dd_batch_diag.py analyze $O/dd_batch_rekeyc.json $D/k2023h2c.parquet $D/k2024.parquet $D/k2025.parquet
$P code/dd_batch_diag.py analyze $O/dd_batch_rekeyc_2026.json $D/k2026c.parquet
$P code/dd_audit.py $O/dd_audit_rekeyc.json $D/k2023h2c.parquet $D/k2024.parquet $D/k2025.parquet $D/k2026c.parquet
$P code/dd_benefit.py $O/dd_benefit_rekeyc.json $D/k2023h2c.parquet $D/k2024.parquet $D/k2025.parquet -- $D/k2026c.parquet
