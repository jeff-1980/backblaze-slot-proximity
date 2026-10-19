#!/bin/bash
# Entry point 3: analyses added during revision (exploratory; all use 2026H1 data except where noted).
# Output goes to results_rerun/revision/. The index-day cohort steps read the daily location runs in results/v6runs/,
# which dd_v6_runs.py builds from the Backblaze daily files (not included).
set -euo pipefail
cd "$(dirname "$0")"; P=${PYTHON:-python3}; D=data; O=results_rerun/revision; R=results/v6runs; mkdir -p $O/v7 $O/v8
K="$D/k2023h2c.parquet $D/k2024.parquet $D/k2025.parquet $D/k2026c.parquet"
# unified run: Tables 1-2, Figures 1-5, Supplementary Tables 1, 3, 5, 9, 12
$P code/dd_unified.py $O/unified_rekeyc $K
$P code/dd_unified_analyse.py $O/unified_rekeyc $O/unified_rekeyc.json $K
# stable-residence subsample, 2026 quarters, model exclusion, Holm (Supplementary Table 1)
$P code/dd_v5_riskset.py $O/v5_riskset $D/k2025.parquet $K
$P code/dd_v5_analyse.py $O/unified_rekeyc $O/v5_riskset $O/v5_review4.json
# failure-level recall and alert cool-down (Supplementary Table 8)
$P code/dd_benefit_perfailure.py $O/dd_benefit_perfailure_rekeyc.json $D/k2023h2c.parquet $D/k2024.parquet $D/k2025.parquet -- $D/k2026c.parquet
$P code/dd_benefit_cooldown.py $O/dd_benefit_cooldown.json $D/k2023h2c.parquet $D/k2024.parquet $D/k2025.parquet -- $D/k2026c.parquet
# index-day cohort (Supplementary Tables 10-14)
PAIRS="$R/runs_k2023h2.parquet $D/k2023h2c.parquet $R/runs_k2024.parquet $D/k2024.parquet $R/runs_k2025.parquet $D/k2025.parquet $R/runs_k2026.parquet $D/k2026c.parquet"
GAP=0 $P code/dd_v6_cohort.py $O/v6_cohort_gap0.json $PAIRS
REPL=1 $P code/dd_v7_cohort.py $O/v7/cohort_main.json $PAIRS
REPL=0 $P code/dd_v7_cohort.py $O/v7/cohort_v6.json $PAIRS
REPL=1 ANCHOR=lastloc $P code/dd_v7_cohort.py $O/v7/cohort_anchor.json $PAIRS
for g in 0 1 3 7; do REPL=1 GAPI=$g B=200 $P code/dd_v7_cohort.py $O/v7/cohort_gi$g.json $PAIRS; done
for g in 0 1 3 7; do REPL=1 GAPE=$g B=200 $P code/dd_v7_cohort.py $O/v7/cohort_ge$g.json $PAIRS; done
$P code/dd_v7_lags.py $O/v7/lags.json $PAIRS
for r in 1 0; do REPL=$r $P code/dd_v8_occupancy.py $O/v8/occupancy_repl$r.json $PAIRS; done
