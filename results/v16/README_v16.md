# results/v16: exploratory physical-pathway tests (manuscript Supplementary Table 18)

Both scripts were written, with their reading rules, before they were run. Everything here is exploratory.

| File | Script | What it is |
|---|---|---|
| `layout.json` | `code/dd_v16_layout.py` | Server-layout test. Index failures in 45- and 60-slot servers, clock F of `dd_v10_clock.py`, NPL = 300 placebo draws (SEED 20261020), joint vault bootstrap B = 1000, windows days 1–30 and 1–7. Assumed layout (not verified from the data): board = slot // 5, row = slot // 15. BP = Mantel–Haenszel ratio of same-board versus neighbouring-board events at slot distances 1–4 in the same row. Run in the container: `NPL=300 python code/dd_v16_layout.py results/v16/layout.json results/v6runs data/k2023h2c.parquet data/k2024.parquet data/k2025.parquet data/k2026c.parquet`. |
| `smart_2024_2025.json` | `code/dd_v16_smart.py` (sha256 0dbc6d42…aa1) | Neighbour event-counter test on 2024–2025: near (distance 1–2) and up to four random far (distance ≥ 16) neighbours of every index failure (seed 20261015); DiD of the share with an increment in days 0–7 after versus the 7 days before, near minus far, vault bootstrap. Counters SMART 191, 189, 192, 12, 199, 188 (raw, not decoded by model, resets not corrected). Read from the Backblaze daily CSVs. |
| `smart_2026H1.json` | `code/dd_v16_smart.py` (same version) | Same statistic on 2026H1; criterion for replicating the electrical signal fixed before the 2026 counters were obtained: DiD(192, 7-day) lower bound > 0. Input: `code/dd_v16_slim_smart.py` parquets of the 2026 Q1/Q2 zips (`--k data/k2026c.parquet`). |
| `smart_pc_log.txt` | — | Log of the 2024–2025 run, including the first attempt that hit the DuckDB spill cap; the I/O-only fix (15 daily files per query) is described in the script docstring. |

The two SMART runs need the Backblaze daily files (with SMART columns), which are not in this repository.
