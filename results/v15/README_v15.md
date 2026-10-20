# results/v15: answers to the mock review of manuscript v14 (exploratory)

Script: `code/dd_v15_review.py` (written before running; a 2-draw test run was checked for errors first). Output: `review.json`.
Settings: clock F of `dd_v10_clock.py`, window days 1–30 unless stated, NPL = 300 placebo draws ((pod, slot) permuted within vault × model × age bin), SEED 20261020, joint vault bootstrap B = 1000, placebo mean held fixed. Every analysis here is exploratory, not pre-registered. Near = slot distance 1–2 (d1), far = distance ≥ 16 (d4), d0 = other drives in the same slot. R = X_near / X_far.

## 0. Reproduction check (clock F reimplementation)

| Group | This run R (95% CI) | unified_rekeyc.json H4 | v10/clock.json F_ALL |
|---|---|---|---|
| Exploration | 1.89 (1.59–2.21); near 245, far 898 | 1.91 (1.63–2.25) | 1.91 (1.62–2.25) |
| Hold-out 2026H1 | 1.97 (1.40–2.64); near 63, far 252 | 2.00 (1.45–2.72) | 1.98 (1.45–2.67) |
| Pooled | 1.91 (1.64–2.20) | 1.93 (1.67–2.23) | — |

The O counts match exactly. E is slightly different from unified: clock F ends a failing drive's exposure at F before the baseline hazard is estimated, so near E is 106.48 against 106.42. The gap in R (−0.014 and −0.028) comes from Monte Carlo error in the placebo mean: 300 draws with a different seed, against 1000 draws in unified. It is about 1.5 Monte Carlo SE. The intervals also differ because the bootstrap seed differs. Exploration placebo means were near 1.143 and far 1.080 (unified: 1.133 and 1.080).

## A1 Leave-one-vault-out (placebo means of the full run held fixed, not re-estimated)

| | Exploration | Hold-out |
|---|---|---|
| R full | 1.894 | 1.973 |
| min / max without one vault | 1.819 (vault 0/1112) / 1.927 (0/1063) | 1.804 (0/1122) / 2.112 (0/1124) |
| largest absolute change | 0.074 | 0.169 |
| distinct vaults (cluster×vault keys) | 333 (846 vault-periods) | 315 |
| distinct vaults with ≥ 1 index failure | 324 | 292 |
| distinct vaults with ≥ 1 near event | **104** | 35 |
| sum over periods of vaults with near event | 136 (24 + 54 + 58) | 35 |

The "136 vaults" in the text is a count of vault-periods: 104 distinct vaults had a near event in exploration.

Five most influential vaults, as cluster/vault (data centre): near events, far events, R without the vault.
- Exploration:
  - 0/1112 (sac2): 11 near, 1 far, R 1.819
  - 0/1128 (sac0): 11, 31, 1.858
  - 0/1022 (sac0): 7, 13, 1.860
  - 0/1063 (phx1): 7, 50, 1.927
  - 0/1137 (sac0): 6, 4, 1.861
- Hold-out:
  - 0/1122 (sac2): 6, 4, 1.804
  - 0/1124 (sac2): 4, 33, 2.112
  - 0/1133 (sac2): 3, 3, 1.886
  - 0/1090 (sac0): 2, 1, 1.909
  - 0/1134 (sac0): 2, 0, 1.909

## A2 Subgroups (subgroup placebo means; vault bootstrap)

Each cell gives R (95% CI) [near O / far O]. A full table with O, E, placebo mean and X for near and far, plus POD and d0, is in `review.json` under `A2_subgroups`. Cells with fewer than 10 near or far events are flagged in the JSON, because their intervals are unreliable.

| Subgroup | Exploration | Hold-out 2026H1 | Pooled (forest plot) |
|---|---|---|---|
| All | 1.89 (1.59–2.21) [245/898] | 1.97 (1.40–2.64) [63/252] | 1.91 (1.64–2.20) [308/1150] |
| 2023H2 | 1.70 (1.02–2.70) [27/114] | | same |
| 2024 | 2.17 (1.66–2.76) [116/428] | | same |
| 2025 | 1.64 (1.26–2.11) [102/356] | | same |
| ams5 | 0.84 (0.45–1.74) [18/40] | 1.21 (0.26–4.00) [4/6]* | 0.88 (0.50–1.60) [22/46] |
| iad1 | 1.48 (0.55–2.55) [9/49]* | 1.88 (0.00–4.96) [4/18]* | 1.58 (0.70–2.72) [13/67] |
| phx1 | 1.78 (1.34–2.45) [49/232] | 2.01 (0.92–3.39) [10/44] | 1.82 (1.40–2.46) [59/276] |
| sac0 | 2.46 (1.87–3.11) [79/249] | 2.21 (0.86–3.87) [14/54] | 2.41 (1.86–2.98) [93/303] |
| sac2 | 2.10 (1.55–2.83) [90/328] | 2.03 (1.20–3.51) [31/130] | 2.08 (1.62–2.73) [121/458] |
| **Cluster 0** | 2.12 (1.77–2.54) [212/784] | **2.19 (1.51–3.13) [53/208]** | 2.13 (1.81–2.52) [265/992] |
| **Other clusters** | 1.08 (0.71–1.58) [33/114] | **1.28 (0.54–2.46) [10/44]** | 1.11 (0.80–1.53) [43/158] |
| Cluster 0, 60-slot | 2.15 (1.80–2.56) [203/769] | 2.24 (1.56–3.22) [53/207] | 2.16 (1.85–2.54) [256/976] |
| Cluster 0, 45-slot | 3.12 (1.29–14.2) [7/12]* | no near event [0/1] | 2.86 (1.19–9.66) [7/13]* |
| Cluster 0, other (26-slot) | 0.68 [2/3]* | no events | 0.77 [2/3]* |
| Other clusters, 60-slot | 1.44 (0.74–2.36) [16/92] | 1.23 (0.21–2.71) [6/42]* | 1.38 (0.82–2.10) [22/134] |
| Other clusters, 45-slot | no servers | no servers | no servers |
| Other clusters, other (26-slot) | 0.99 (0.52–2.32) [17/22] | 2.18 [4/2]* | 1.10 (0.61–2.64) [21/24] |
| All 60-slot | 2.07 (1.74–2.46) [219/861] | 2.06 (1.45–2.84) [59/249] | 2.07 (1.78–2.41) [278/1110] |
| All 45-slot | 3.12 [7/12]* | 0 near [0/1]* | 2.86 [7/13]* |
| All other | 0.94 (0.52–2.05) [19/25] | 2.48 [4/2]* | 1.05 (0.63–2.28) [23/27] |

\* Fewer than 10 near or far events: the interval is not meaningful. The 2025–2026 data centre yyz1 and the drives with no data centre had no near events.

Capacity class follows `dd_sample_flow.py`: the smallest of 26, 45 and 60 that is at least the highest slot number seen in the server plus 1. The class is that of the index drive's server, and under the placebo it is that of the server the index drive is placed in. All 45-slot servers are in cluster 0. In the other clusters, the "other" class is the 26-slot servers.

## A3 Service-visit anchor (placebo means of the full-window run)

Rrep is the first day another drive is recorded in the index drive's own slot after its last located day (`dd_v10_clock.lr`). Index failures by where Rrep falls:

| Group | On or before day 1 | Days 2–30 | None in window | Rrep missing |
|---|---|---|---|---|
| Exploration (10,781) | 6,315 | 4,377 | 89 | 81 |
| Hold-out (2,483) | 1,526 | 844 | 113 | 78 |

In most index failures, therefore, the replacement is already recorded when the window opens.

| Part of the window | Exploration R [near/far O] | Hold-out R [near/far O] |
|---|---|---|
| Before Rrep (including no replacement) | 1.79 (0.67–2.92) [10/40] | 1.39 (0.54–2.68) [9/50] |
| On/after Rrep | 1.90 (1.60–2.23) [235/858] | 2.12 (1.47–2.94) [54/202] |
| No replacement in window only | 2.30 [2/5]* | 1.82 [2/9]* |
| Index failures with Rrep in window, part before Rrep | 1.69 (0.53–3.11) [8/35] | 1.27 (0.40–2.54) [7/41] |

Before Rrep in the hold-out, X_near is 2.75 and X_far is 1.98, so far events are also raised before the replacement.

Follow-on date minus Rrep, in days:

| | Exploration near | Exploration far | Hold-out near | Hold-out far |
|---|---|---|---|---|
| Median | 9 | 11 | 9.5 | 5 |
| IQR | 3–17 | 4–20 | 0–22.75 | 0–18 |
| Share 0–3 days | 24.3% | 20.2% | 30.6% | 24.5% |
| Share before Rrep | 3.3% | 3.9% | 12.9% | 18.9% |

Server proxy Srv is the first new drive anywhere in the index server after the last located day:

| | Exploration R [near/far O] | Hold-out R [near/far O] |
|---|---|---|
| Window before Srv | 1.53 (0.46–3.14) [6/29] | 1.17 (0.43–2.49) [7/46] |
| On/after Srv | 1.90 (1.59–2.23) [239/869] | 2.15 (1.53–2.94) [56/206] |
| Index failures with a server event in the window | 1.90 (1.59–2.21) [245/897] | 1.98 (1.41–2.67) [61/243] |
| Index failures without one | 28 indices, no near events | 1.78 [2/9]* (112 indices) |

Reading: in these data a slot or server arrival is recorded almost at once after nearly every failure, so the time before service is short. Only 10 (exploration) and 9 (hold-out) near events fall before it, and the before/after contrast cannot be estimated with useful precision. The point estimates before Rrep are lower than after it, but the intervals overlap widely. The timing distributions of near and far events relative to Rrep are similar.

## A4 Early life

(a) Age at failure is power-on hours at first appearance in the period divided by 24, plus the calendar days from first record to F. poh0 was available for every counted follow-on.

| | Median days (IQR) | Share < 90 days |
|---|---|---|
| Exploration near | 1,754 (1,236–2,106) | 3.7% |
| Exploration far | 1,714 (1,237–2,090) | 2.6% |
| Exploration d0 | 22 (11–1,114) | 65.6% |
| Hold-out near | 2,281 (1,718–2,492) | 1.6% |
| Hold-out far | 2,279 (1,714–2,504) | 1.2% |
| Hold-out d0 | 10 (6–14) | 88.2% |

Days since first record in the period are also in the JSON, but they are left-censored at the period start.

(b) and (c): each has its own placebo run. In (b), the index failures are unchanged.

| | Base | (b) exclude first 90 days of service | (c) first age bin split 0–30 / 31–90 / 91–182 d |
|---|---|---|---|
| Exploration R | 1.89 (1.59–2.21) | 1.90 (1.60–2.22) | 1.92 (1.61–2.25) |
| Exploration X near | 2.01 (1.74–2.31) [245] | 2.02 (1.75–2.30) [236] | 2.03 (1.76–2.33) [245] |
| Exploration X far | 1.06 (0.98–1.15) [898] | 1.06 (0.97–1.16) [875] | 1.06 (0.97–1.15) [898] |
| Exploration X d0 | 2.90 (2.22–3.63) [96] | 5.61 (3.11–10.5) [33] | 2.01 (1.57–2.48) [96] |
| Hold-out R | 1.97 (1.40–2.64) | 1.98 (1.41–2.64) | 1.98 (1.40–2.65) |
| Hold-out X near | 2.32 (1.73–2.91) [63] | 2.33 (1.71–2.93) [62] | 2.35 (1.75–2.96) [63] |
| Hold-out X far | 1.18 (0.95–1.38) [252] | 1.18 (0.96–1.37) [249] | 1.19 (0.96–1.39) [252] |
| Hold-out X d0 | 1.84 (1.14–2.76) [17] | 3.92 (0.00–21.6) [2] | 1.46 (0.92–2.15) [17] |

R does not depend on early-life drives. The distance-0 excess is mostly failures of young replacement drives. Splitting the first age bin lowers d0 X to 2.01 in exploration and 1.46 in the hold-out, where the interval includes 1. Removing drives in their first 90 days leaves only 33 and 2 d0 events.

The d0 X here (2.90) is higher than in unified (2.65): its placebo mean is noisy (300 draws) and clock F changes E.

## A5 Absolute scale (per 1,000 index failures)

n_index is the number of index failures whose window starts in the period: 10,779 in exploration and 2,468 in the hold-out.

| | O / 1,000 | E / 1,000 | E × placebo / 1,000 | Excess / 1,000 (95% CI) |
|---|---|---|---|---|
| Exploration near, 1–30 d | 22.7 | 9.88 | 11.30 | **11.4 (8.1–14.8)** |
| Exploration far, 1–30 d | 83.3 | 72.6 | 78.4 | 4.9 (−1.9 to 11.6) |
| Exploration near, 1–3 d | 4.17 | 1.02 | 1.26 | 2.9 (1.6–4.5) |
| Exploration far, 1–3 d | 10.9 | 7.56 | 9.04 | 1.8 (−0.2 to 3.9) |
| Hold-out near, 1–30 d | 25.5 | 9.87 | 10.98 | **14.5 (7.5–21.2)** |
| Hold-out far, 1–30 d | 102.1 | 76.6 | 86.6 | 15.5 (−4.0 to 33.5) |
| Hold-out near, 1–3 d | 6.48 | 1.05 | 1.57 | 4.9 (1.8–8.3) |
| Hold-out far, 1–3 d | 27.1 | 8.21 | 12.9 | 14.3 (7.5–21.5) |

Double counting: one follow-on failure counts once for every index window, in its server, that contains it. Counted events and distinct follow-on drives:

| | Near events / distinct drives (drives counted more than once) | Far events / distinct drives |
|---|---|---|
| Exploration 1–30 d | 245 / 232 (11) | 898 / 821 |
| Hold-out 1–30 d | 63 / 58 (5) | 252 / 205 |
| Exploration 1–3 d | 45 / 44 | 117 / 114 |
| Hold-out 1–3 d | 16 / 16 | 67 / 63 |

## A6 Provenance

(i) **232 and 58 distinct drives.** These counts are correct and archived in `results/review2_explore.json` and `results/review2_2026.json` (`A_raw.d1.unique_followers` = 232 and 58; `followers_multi_index` = 11 and 5). This run recomputes them exactly. In the hold-out, the same-batch / different-batch near counts are also 58 / 5 (`dd_batch_rekeyc_2026.json` POD_d1_SB.O = 58, POD_d1_OT.O = 5; `unified_rekeyc.json` H7 Oa = 58). The match is a coincidence: the distinct-drive count has its own source. The audit's AUD-12 "no source" finding is resolved.

(ii) **Index-failure totals.**
- 10,061: failures of drives with a slot number and a complete 30-day window (F ≤ period end − 30). Sources: `review2_explore.json` n_index.FULL, and unified n_index_full 1,723 + 4,198 + 4,140.
- 10,060: `v7/lags.json` counts 1,723 + 4,197 + 4,140. In 2024, one failing drive has no located daily record on or before its failure (`n_no_located_record_before_failure` = 1).
- 706 vs 705: there are 706 replacement-blocked hold-out indices (`v7/cohort_main.json` index_by_type.ext_blocked). Re-running `dd_v7_cohort.load` shows that 705 of them have a distance-0 exposure. The exception is index 6210A18BFVKG, which failed on day 68. Its replacement 52U0A043FVKG was recorded only on day 68 and failed that same day, so the rule "not failing on the index day" excludes it. All 705 d0 exposures come from blocked indices.
- 10,779 and 2,468 (clock tables): failures whose window starts in the period (t + 1 ≤ period end). The totals 10,781 and 2,483 are all failures.

(iii) **Drive-days.** 333,732,420 is the number of daily rows with a server (vault, pod) location: 48,392,985 + 105,379,761 + 117,394,187 + 62,565,487. The sources are `dd_idcheck_rekeyc_2.json` and, for 2023H2, `dd_review2_pc_master.txt` line 2. Drive-days with a slot number total 328,978,223 (`v6runs/runs_k*.json` rows). Distinct slot location-days total 328,978,216. The difference of 7 is the seven 2025 location-days that held two drives. The text should say "333.7 M drive-days with a server location, of which 329.0 M had a slot number".

(iv) **Pre-specified drive-snapshot metric** (`results/dd_benefit_rekeyc.json`; 6,164,654 test rows, 8,065 positives).

| | Baseline M0 | Neighbour model M1 | Difference (95% CI) |
|---|---|---|---|
| AUC | 0.6707 | 0.6724 | |
| Capture at 0.5% | 2.02% | 2.18% | +0.16 pp (−0.46 to 1.16) |
| Capture at 1% | 3.87% | 3.72% | −0.15 pp (−0.68 to 1.39) |
| Capture at 2% | 8.73% | 8.34% | −0.38 pp (−0.71 to 1.43) |

Capture is the share of positive rows that fall within the budget. The file's reading is "LOCAL_MARKER_ONLY".

(v) **Random tie order.** Source: `results/dd_benefit_cooldown.json`, `budgets.<b>.tie_random`, 20 random tie orders. s0 is M0 and s1 is M1; c is the cool-down in weekly scans. Caught failures:

| Budget, cool-down | M0 row order | M0 random ties (mean) | M1 row order | M1 random ties (mean) |
|---|---|---|---|---|
| 1%, c = 0 | 80 | 92–98 (95.9) | 117 | 114–118 (116.1) |
| 2%, c = 0 | 188 | 198–203 (200.9) | 192 | 191–195 (192.4) |
| 1%, c = 1 | 181 | 186–195 | 183 | 182–185 |
| 0.5%, c = 0 | 42 | 47–49 | 57 | 61–68 |

At 2% the sign flips: M1 is ahead under row order and M0 is ahead under random ties. The c = 1 comparison at 1% also flips.

(vi) **Original-key columns of Supplementary Table 2.** No result file in `results/` or `rerun_check/` holds these values. They come from the original frozen run of 2026-10-14, whose output files (dd_confirm_2026.json, dd_batch_diag_2026.json) were not archived. The only archived original-key frozen-script output is `results/rerun_check/dd_confirm_origkey_2026-10-19.json`: 2026H1 only, on the current tables, with POD X 1.288, d1 2.192, d4 1.187 and tome 0.928.
