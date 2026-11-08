# Grouped BCE validation audit

## Root cause

The historical grouped report selected the row with disease == LABEL_NAMES[0] and displayed its per-disease `bce` under the column name 'Overall BCE'. `grouped_metrics` did not expose a group-level macro BCE, so reports accidentally used column 0 of the ten-class vector.

## Fix

grouped_metrics now writes macro_bce / macro_positive_bce / macro_negative_bce = ten-class mean within each group; all report tables use macro_bce and never a single disease row.

## Assertions

| Experiment | Split | Single N | Multi N | Single Macro-BCE | Multi Macro-BCE | Weighted | Overall | Passed |
|---|---|---:|---:|---:|---:|---:|---:|---|
| H0 | train | 8507 | 10006 | 0.298295 | 0.183286 | 0.236135 | 0.236135 | True |
| H0 | valid | 391 | 468 | 0.392367 | 0.243360 | 0.311185 | 0.311185 | True |
| H2 | train | 8507 | 10006 | 0.289079 | 0.188996 | 0.234986 | 0.234986 | True |
| H2 | valid | 391 | 468 | 0.392892 | 0.242761 | 0.311098 | 0.311098 | True |
| H2B | train | 8507 | 10006 | 0.293203 | 0.184864 | 0.234647 | 0.234647 | True |
| H2B | valid | 391 | 468 | 0.393664 | 0.242474 | 0.311292 | 0.311292 | True |
| H2C | train | 8507 | 10006 | 0.227297 | 0.165772 | 0.194044 | 0.194044 | True |
| H2C | valid | 391 | 468 | 0.419722 | 0.257130 | 0.331139 | 0.331139 | True |
| H4 | train | 8507 | 10006 | 0.302577 | 0.186215 | 0.239685 | 0.239685 | True |
| H4 | valid | 391 | 468 | 0.396813 | 0.243054 | 0.313042 | 0.313042 | True |

- 同时断言 `BCE_overall = mean_k BCE_k`（十类均值）；所有断言通过，浮点误差 < 1e-6（BF16/float32 推理口径）。

## Old H3-B reported values vs corrected macro BCE

旧报告列名是 “Overall BCE”，但数值来自行 `disease == LABEL_NAMES[0]`（Enlarged Cardiomediastinum），不是十类均值；H3-B 文件未覆盖，本目录输出修正版。旧值与第 0 类 BCE 一致：

| Group (H3-B, valid) | Old reported 'Overall BCE' |
|---|---:|
| AP-only | 0.155770 |
| PA+Lateral | 0.077841 |
| Single-image | 0.168390 |
| Multi-image | 0.084929 |

| Group (H3-B, valid) | Old reported | Corrected macro BCE |
|---|---:|---:|
| AP-only | 0.155770 | 0.388899 |
| PA+Lateral | 0.077841 | 0.223740 |
| Single-image | 0.168390 | 0.392849 |
| Multi-image | 0.084929 | 0.251815 |

旧值等于该组第 0 类 BCE（Enlarged Cardiomediastinum）：{'AP-only': True, 'PA+Lateral': True, 'Single-image': True, 'Multi-image': True}。

对照：H3-B valid 十类总体 BCE 为 0.332165；旧值的 AP-only 0.155770 等明显是该病种 BCE，而不是十类均值。修正版见 `grouped_performance_comparison.csv` 的 `macro_bce` 列。

## AUROC/AUPRC audit

- 分组 `macro_auroc`/`macro_auprc` 直接来自 `study_metrics` 的十类宏平均（只对有效类别求平均），不是第 0 类；未发现同类索引错误。
- 逐病种值仍保留在 `bce`/`auroc`/`auprc` 列，报告只使用 `macro_bce`。

