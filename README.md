# Slot-adjacent drive failures: code, derived data and results

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23486544.svg)](https://doi.org/10.5281/zenodo.23486544)

Code, derived per-drive tables and result files for the manuscript "Slot-adjacent drive failures cluster within servers, with heterogeneous effects across a public storage fleet".

The analysis uses the public [Backblaze Drive Stats](https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data) daily files from 2023-07-01 to 2026-06-30. It asks whether, after a drive fails, other drives in the same server fail more often than shared model, age and vault would predict, and whether the excess depends on slot-number distance.

## Contents

| Folder | What it holds |
|---|---|
| `data/` | One row per drive per period: `k2023h2c`, `k2024`, `k2025` (exploration) and `k2026c` (2026H1, retrospective time hold-out). |
| `code/` | Analysis scripts in the versions that produced `results/`. `CHANGELOG_io_fixes.txt` lists every change made after the freeze; none changes analysis logic. |
| `preregistration/` | The scripts and confirmation addendum frozen on 2026-10-14 17:31 UTC, before the 2026 files were downloaded, with SHA-256 hashes in `FROZEN.txt`. |
| `results/` | JSON output of each step. The two large intermediate arrays (`unified_rekeyc.npz`, `v5_riskset.npz`, 84 MB) are not in the repository; `run_revision.sh` rebuilds them. `results/v6runs/` holds the daily location runs used by the index-day cohort; `results/v7/` and `results/v8/` hold the cohort and occupancy audits, and `results/v10/` the record-clock audit. |
| `figures/` | Figures of the manuscript: `fig1_schematic.png` is Figure 1, and `fig1_sets`, `fig2_distance`, `fig3_timecourse`, `fig4_batch` and `fig5_generalisation` are Figures 2–6 (file names keep the numbering of `code/make_figs_v8.py`). |

## Server key

Backblaze reuses `vault_id` across clusters, so a server is `(cluster_id, vault_id, pod_id)`. In the tables, `vault = cluster * 100000 + vault_raw`, so every script that groups by `vault` and `pod` uses the corrected key. The frozen plan described the key as `(vault_id, pod_id)`; `code/dd_original_key.py` rebuilds the tables under that original key.

## Data columns

`serial_number, model, cluster, datacenter, vault_raw, pod, slot, n_pods_seen, first, last, fail, fdate, poh0, vault, period_start, period_end, n_files`.

- Location fields are each drive's last non-null value in the period.
- `poh0` is power-on hours (SMART 9) at first appearance.
- `fail` and `fdate` follow the Backblaze failure label (`fdate` = first failure record).

## Reproduce

Requirements: Python 3.11+ and `pip install -r requirements.txt` (package versions are listed there). Everything runs on a CPU, and output goes to `results_rerun/`.

| Entry point | What it reruns | Uses 2026H1 | Result files (compare with `results/`) |
|---|---|---|---|
| `bash run_frozen.sh` | The frozen scripts in `preregistration/` under the original key (vault_id, pod_id) | Yes, in the confirm and `_2026` runs | `frozen_origkey/dd_explore_origkey.json`, `dd_confirm_origkey.json`, `dd_batch_origkey(_2026).json` (Supplementary Table 2, original-key columns) |
| `bash run_corrected.sh` | The frozen hypotheses, and the scripts written before they were run (audit, operational test), under the corrected key; sample flow | Yes, in the confirm, `_2026`, audit and operational runs | `corrected/dd_explore_rekeyc.json`, `dd_confirm_rekeyc.json`, `dd_batch_rekeyc(_2026).json`, `dd_audit_rekeyc.json`, `dd_benefit_rekeyc.json`, `sample_flow.json` (Supplementary Tables 1, 2 and 6) |
| `bash run_revision.sh` | Analyses added during revision, all exploratory | Yes | `revision/unified_rekeyc.json` (Tables 1–2, Figures 2–6, Supplementary Tables 1, 3, 5, 9 and 12), `v5_review4.json`, `dd_benefit_perfailure_rekeyc.json`, `dd_benefit_cooldown.json` (Supplementary Table 8), `v6_cohort_gap0.json`, `v7/cohort_*.json` and `v7/lags.json` (Supplementary Tables 10, 11 and 14), `v8/occupancy_repl{1,0}.json` (Supplementary Table 13), `v10/clock.json` (Supplementary Table 15) |

The manuscript's "Supplementary information" table maps every table to its script and result file. In our runs, the corrected-key confirmation step reproduced `results/dd_confirm_rekeyc.json` exactly, and `dd_sample_flow.py` reproduced `results/sample_flow.json` exactly. The original-key results in Supplementary Table 2 were first computed on tables built before the clean re-download of 17 October 2026, so `run_frozen.sh` can differ from them slightly. A rerun of the frozen confirmation step under the original key on the current tables (`results/rerun_check/dd_confirm_origkey_2026-10-19.json`) gave H1 1.29 (lower bound 1.14) and H4 1.85 (1.33–2.50), against 1.28 (1.13–1.45) and 1.90 (1.37–2.58) in Supplementary Table 2; every verdict was the same.

### Steps that need the Backblaze daily files

These are not covered by the three entry points:

1. Download the quarterly zips from Backblaze and reduce each one to a parquet file: `python code/dd_slim_k.py <zip> <out.parquet> <tmpdir>`.
2. Build each period's table: `python code/dd_rekey.py csv <dir> <start> <end> <out.parquet>`.
3. Build the daily location runs for the index-day cohort: `python code/dd_v6_runs.py <out_dir> <label> <src_dir> <start> <end> [...]`. The output is shipped in `results/v6runs/`.
4. Run the daily audits and exploratory mechanism tests:
   - `dd_overlap_check.py`, `dd_residency.py` and `dd_dup_diag.py`: daily location audits;
   - `dd_idcheck.py`: Supplementary Table 4;
   - `dd_mechanism.py` part A: slot-distance profile and replacement timing;
   - `dd_mechanism.py` part B: SMART panel;
   - `dd_mech_C.py`: service-visit test (runs on `data/`).

The shared-link test is part of `dd_audit.py`, which `run_corrected.sh` runs.

## Licences

- Code: MIT (`LICENSE`).
- Derived tables, results and figures: CC BY 4.0 (`LICENSE-DATA`).
- The source data are Backblaze Drive Stats. Cite Backblaze as the source and do not sell the data, as Backblaze's terms require.

## Citation

See `CITATION.cff`. Release v1.1.0 is archived at Zenodo: [https://doi.org/10.5281/zenodo.23486544](https://doi.org/10.5281/zenodo.23486544). Release v1.0.0: [https://doi.org/10.5281/zenodo.23478229](https://doi.org/10.5281/zenodo.23478229).
