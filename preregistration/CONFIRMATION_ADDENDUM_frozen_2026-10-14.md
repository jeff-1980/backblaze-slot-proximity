# 2026 年确认阶段：冻结的判据（2026-10-14 写定，早于打开任何 2026 数据）

在 dd_contagion.py 文件头的 HYPOTHESES 后追加以下内容（探索结果见 notes/deepdive_backblaze_explore_2026-10-14.md）。

ADDENDUM (frozen 2026-10-14 after exploration on 2023H2–2025 and BEFORE opening any 2026 file):
  H1–H5 are tested on 2026 unchanged. H5 failed in exploration and is kept only for completeness.
  H6 (post-exploration): TOME excess over days 8–30, X_TOME(8-30), > 1 with CI lower bound > 1.
     Implemented by adding window "M" = (8, 30) to WINS. Adding a window does not change the RNG stream.
  H7 (batch control, post-exploration, via dd_batch_diag.py): among SAMEBATCH pod pairs only
     (same model and |poh0 difference| < 720 h), the X ratio of d1 to d4 is > 1 with CI lower bound > 1.
  Confirmation data = all 2026 daily files Backblaze has published, built as one period
     (2026-01-01 to the last available date). Same seed, 100 placebos, 1000-draw vault bootstrap.
  If 2026 has fewer than 150 index failures, the results are underpowered and none of H1–H7 is called
     confirmed or refuted.
  Headline claims for a paper = the hypotheses among H1, H4, H6, H7 that are confirmed.
  Exploration finding that motivated H7: d1/d4 = 1.85 [1.54, 2.18] among SAMEBATCH pairs,
     2.18 [1.27, 3.44] among OTHER pairs; the SAMEBATCH fraction is about 0.71 at every distance.
