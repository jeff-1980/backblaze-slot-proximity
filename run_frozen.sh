#!/bin/bash
# Entry point 1: the frozen analysis (preregistration/, hashes in FROZEN.txt) under the original server key
# (vault_id, pod_id) of the frozen plan. Output goes to results_rerun/frozen_origkey/.
# Uses 2026H1 data in the two *_2026 / confirm runs.
set -euo pipefail
cd "$(dirname "$0")"; P=${PYTHON:-python3}; O=results_rerun/frozen_origkey; D=$O/data; mkdir -p $O
$P code/dd_original_key.py $D data/k2023h2c.parquet data/k2024.parquet data/k2025.parquet data/k2026c.parquet
$P preregistration/dd_contagion.py analyze $O/dd_explore_origkey.json $D/k2023h2c.parquet $D/k2024.parquet $D/k2025.parquet
$P preregistration/dd_contagion.py analyze $O/dd_confirm_origkey.json $D/k2026c.parquet
$P preregistration/dd_batch_diag.py analyze $O/dd_batch_origkey.json $D/k2023h2c.parquet $D/k2024.parquet $D/k2025.parquet
$P preregistration/dd_batch_diag.py analyze $O/dd_batch_origkey_2026.json $D/k2026c.parquet
