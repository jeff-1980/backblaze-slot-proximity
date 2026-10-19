#!/bin/bash
# Entry point 2: the frozen hypotheses and the scripts written before they were run (audit, operational test),
# under the corrected server key (cluster, vault, pod). Output goes to results_rerun/corrected/.
# Expected: dd_explore_rekeyc.json, dd_confirm_rekeyc.json (2026H1), dd_batch_rekeyc(_2026).json, dd_audit_rekeyc.json,
# dd_benefit_rekeyc.json (trained on 2023H2-2025, tested on 2026H1), sample_flow.json.
set -euo pipefail
cd "$(dirname "$0")"; P=${PYTHON:-python3}; D=data; O=results_rerun/corrected; mkdir -p $O
K="$D/k2023h2c.parquet $D/k2024.parquet $D/k2025.parquet"
$P code/dd_overlap_check.py $K $D/k2026c.parquet
$P code/dd_contagion.py analyze $O/dd_explore_rekeyc.json $K
$P code/dd_contagion.py analyze $O/dd_confirm_rekeyc.json $D/k2026c.parquet
$P code/dd_batch_diag.py analyze $O/dd_batch_rekeyc.json $K
$P code/dd_batch_diag.py analyze $O/dd_batch_rekeyc_2026.json $D/k2026c.parquet
$P code/dd_audit.py $O/dd_audit_rekeyc.json $K $D/k2026c.parquet
$P code/dd_benefit.py $O/dd_benefit_rekeyc.json $K -- $D/k2026c.parquet
$P code/dd_sample_flow.py $O/sample_flow.json $K $D/k2026c.parquet
