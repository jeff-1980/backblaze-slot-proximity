# results/v17: corrections and sensitivity analyses after the 2026-10-24 mock review

| File | Script | What it is |
|---|---|---|
| `dd_benefit_rekeyc_auc.json`, `benefit_log.txt` | `code/dd_benefit.py` | Rerun after the AUC fix (tied scores now get average ranks; see `code/CHANGELOG_io_fixes.txt`). AUC M0 0.67068, M1 0.67244 (archived `results/dd_benefit_rekeyc.json`: 0.67070, 0.67243). All other outputs identical to the archived file. |
| `dd_benefit_perfailure_auc.json`, `perfailure_log.txt` | `code/dd_benefit_perfailure.py` | Same fix and rerun. Same AUC; failure-level recall, intervals and coverage identical to `results/dd_benefit_perfailure_rekeyc.json` (this rerun also contains the alert-burden fields added on 2026-10-18). |
| `slotchange.json`, `slotchange_log.txt` | `code/dd_v17_slotchange.py` | Drives recorded at more than one slot of the same server within a period (from `results/v6runs`): 1, 42, 28 and 676 in 2023H2, 2024, 2025 and 2026H1. R (days 1–30, clock F, NPL = 300, SEED 20261024, vault bootstrap B = 1000) with and without these drives as index failures and near/far members: exploration 1.93 → 1.93, hold-out 2.01 → 1.98. One near event involved such a drive. Exploratory. |

Commands: `python code/dd_benefit.py results/v17/dd_benefit_rekeyc_auc.json data/k2023h2c.parquet data/k2024.parquet data/k2025.parquet -- data/k2026c.parquet`; the same arguments for `dd_benefit_perfailure.py`; `NPL=300 python code/dd_v17_slotchange.py results/v17/slotchange.json results/v6runs data/k2023h2c.parquet data/k2024.parquet data/k2025.parquet data/k2026c.parquet`.
