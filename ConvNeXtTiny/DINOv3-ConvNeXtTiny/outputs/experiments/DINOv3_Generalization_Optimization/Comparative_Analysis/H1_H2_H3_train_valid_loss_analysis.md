# H1 / H2 / H3 Train–Validation Loss Analysis

本报告基于官方 Train（18,513 Study / 30,344 图像 / 11,055 Subject）与 Validation（859 Study / 1,422 图像 / 228 Subject）的**完整确定性无增强推理**。所有数字来自各自 Best / Last Checkpoint；Test 标签从未读取。

## 0. 实验与统计口径

- H0: `{"mode": "baseline"}`
- H1: `{"mode": "ap_sampling", "target_ap_fraction": 0.55, "ap_weight": 1.2}`
- H2: `{"mode": "single_study_weight", "single_weight": 1.25}`
- H3: `{"mode": "l2sp"}`

- 原始 BCE（raw Study BCE）定义：对每个 Study 先对 10 类 BCE 取平均，再对全部 Study 取平均；训练/验证均使用该未加权口径，不受 H1 采样、H2 单图权重或 H3 L2-SP 项影响。
- 各实验实际优化的目标：H0/H1 为原始未加权 Study BCE；H2 为归一化单图加权 BCE（记录于 `training_history.csv` 的 `weighted_objective_bce`）；H3 为原始 BCE + λ_SP·R_SP（记录于 `objective_loss`）。比较表格只使用原始 BCE。
- 程序化校验：逐类 BCE 的十类平均与按 Study 平均的 BCE 在全部实验/切分上一致（两者在同一批全有效标签上数学等价；分析代码同时输出 `overall_bce` 与 `study_level_bce`）。
- H0 为 R5 Seed=42 基线复用（配置一致性检查通过）；H0 逐 Epoch 的十类训练批次 BCE 历史未记录，已标记为不可用；其 Validation 逐 Epoch 逐类 BCE 由历史保存的 逐 Epoch 验证预测恢复，Best/Last 的完整确定性逐类 BCE 已重新计算。
- 所有模型均从同一原始 DINOv3 预训练权重独立训练，仅改变指定干预；H1/H2/H3 未叠加。
- **H3 收敛性说明**：H3 的三条件早停未触发，训练在 epoch 12 因总实验时间预算停止，停止时 Validation AUROC 仍在缓慢上升；因此 H3 不是收敛结果，其与 H0 的比较只能作为方向性证据，不能据此断言 L2-SP 在完整 20-epoch 预算下的最终效果。

## A. 总体 Loss 与 AUROC 对照（Best Checkpoint）

| Experiment | Train Overall BCE | Valid Overall BCE | BCE Gap | Valid Macro-AUROC | Valid Macro-AUPRC | Best Epoch | Completed / Stop |
|---|---:|---:|---:|---:|---:|---:|---|
| H0 | 0.236135 | 0.311185 | +0.075051 | 0.803469 | 0.428405 | 6 | 9 / three_condition_early_stop |
| H1 | 0.266316 | 0.329965 | +0.063649 | 0.785835 | 0.388620 | 4 | 7 / three_condition_early_stop |
| H2 | 0.234986 | 0.311098 | +0.076112 | 0.805038 | 0.424958 | 6 | 9 / three_condition_early_stop |
| H3 | 0.315458 | 0.332165 | +0.016706 | 0.763786 | 0.372637 | 12 | 12 / resource_budget_stop |

- Train BCE 最低：**H2**（0.234986）。
- Validation BCE 最低：**H2**（0.311098）。
- Validation Macro-AUROC 最高：**H2**（0.805038）。
- Loss 与 AUROC 的变化方向不一定一致：BCE 同时衡量概率校准与排序，而 AUROC 只衡量排序。下文逐项检查是否存在 Train Loss 上升/Valid Loss 下降、Valid Loss 下降/AUROC 不升、Valid Loss 上升/AUROC 上升。

### Last Checkpoint（参考）

| Experiment | Train Overall BCE | Valid Overall BCE | Valid Macro-AUROC | Best Epoch |
|---|---:|---:|---:|---:|
| H0 | 0.097966 | 0.453414 | 0.775794 | 6 |
| H1 | 0.153405 | 0.438363 | 0.755185 | 4 |
| H2 | 0.097784 | 0.448960 | 0.765304 | 6 |
| H3 | 0.315458 | 0.332165 | 0.763786 | 12 |

## B. 十类疾病 Train / Valid BCE 对照（Best Checkpoint，实测）

| Disease | H0 Train | H0 Valid | H1 Train | H1 Valid | H2 Train | H2 Valid | H3 Train | H3 Valid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Enlarged Cardiomediastinum | 0.125810 | 0.124876 | 0.125176 | 0.118915 | 0.122395 | 0.120108 | 0.137533 | 0.121995 |
| Pneumothorax | 0.108250 | 0.141200 | 0.130627 | 0.159182 | 0.104915 | 0.138153 | 0.162515 | 0.157552 |
| Consolidation | 0.150230 | 0.178681 | 0.158094 | 0.189182 | 0.149358 | 0.178796 | 0.175162 | 0.190039 |
| Pneumonia | 0.216491 | 0.256572 | 0.231947 | 0.265410 | 0.218036 | 0.259797 | 0.258776 | 0.268491 |
| Edema | 0.204797 | 0.276348 | 0.266499 | 0.329877 | 0.201649 | 0.277488 | 0.278735 | 0.283880 |
| Cardiomegaly | 0.325708 | 0.422168 | 0.363371 | 0.455875 | 0.325988 | 0.421395 | 0.422396 | 0.443446 |
| Atelectasis | 0.337646 | 0.434288 | 0.366343 | 0.449596 | 0.335921 | 0.433796 | 0.434993 | 0.472142 |
| Lung Opacity | 0.411262 | 0.503647 | 0.433951 | 0.512266 | 0.410365 | 0.505755 | 0.491505 | 0.513734 |
| Pleural Effusion | 0.220422 | 0.357672 | 0.264435 | 0.370529 | 0.220700 | 0.355403 | 0.366441 | 0.418887 |
| No Finding | 0.260731 | 0.416401 | 0.322718 | 0.448815 | 0.260532 | 0.420283 | 0.426526 | 0.451482 |
| Overall BCE | 0.236135 | 0.311185 | 0.266316 | 0.329965 | 0.234986 | 0.311098 | 0.315458 | 0.332165 |

### 最大 Train→Valid BCE Gap（跨实验，Top 10）

| Disease | Experiment | Valid−Train Gap |
|---|---|---:|
| No Finding | H2 | +0.159751 |
| No Finding | H0 | +0.155670 |
| Pleural Effusion | H0 | +0.137250 |
| Pleural Effusion | H2 | +0.134702 |
| No Finding | H1 | +0.126097 |
| Pleural Effusion | H1 | +0.106094 |
| Atelectasis | H2 | +0.097876 |
| Atelectasis | H0 | +0.096643 |
| Cardiomegaly | H0 | +0.096460 |
| Cardiomegaly | H2 | +0.095407 |

### Valid BCE 相对 H0 的改善（Best）

| Disease | H1 Δ | H2 Δ | H3 Δ |
|---|---:|---:|---:|
| Enlarged Cardiomediastinum | -0.005961 | -0.004767 | -0.002881 |
| Pneumothorax | +0.017982 | -0.003047 | +0.016352 |
| Consolidation | +0.010502 | +0.000116 | +0.011359 |
| Pneumonia | +0.008838 | +0.003225 | +0.011919 |
| Edema | +0.053529 | +0.001141 | +0.007532 |
| Cardiomegaly | +0.033707 | -0.000774 | +0.021277 |
| Atelectasis | +0.015308 | -0.000492 | +0.037854 |
| Lung Opacity | +0.008619 | +0.002108 | +0.010087 |
| Pleural Effusion | +0.012856 | -0.002270 | +0.061215 |
| No Finding | +0.032414 | +0.003882 | +0.035081 |
| Overall BCE | +0.018779 | -0.000088 | +0.020979 |

## C. 十类疾病 Positive / Negative BCE（Best Checkpoint）

| Disease | Exp | Valid Positive BCE | Valid Negative BCE | Pos/Study | Neg/Study |
|---|---|---:|---:|---:|---:|
| Enlarged Cardiomediastinum | H0 | 2.667079 | 0.051806 | 24 | 835 |
| Enlarged Cardiomediastinum | H1 | 3.273206 | 0.028253 | 24 | 835 |
| Enlarged Cardiomediastinum | H2 | 3.008307 | 0.037094 | 24 | 835 |
| Enlarged Cardiomediastinum | H3 | 3.064595 | 0.037417 | 24 | 835 |
| Pneumothorax | H0 | 1.678906 | 0.071985 | 37 | 822 |
| Pneumothorax | H1 | 2.850037 | 0.038061 | 37 | 822 |
| Pneumothorax | H2 | 1.841394 | 0.061487 | 37 | 822 |
| Pneumothorax | H3 | 2.544403 | 0.050115 | 37 | 822 |
| Consolidation | H0 | 2.497378 | 0.047487 | 46 | 813 |
| Consolidation | H1 | 2.725879 | 0.045655 | 46 | 813 |
| Consolidation | H2 | 2.562060 | 0.043950 | 46 | 813 |
| Consolidation | H3 | 2.487952 | 0.060022 | 46 | 813 |
| Pneumonia | H0 | 2.609858 | 0.054267 | 68 | 791 |
| Pneumonia | H1 | 2.055618 | 0.111511 | 68 | 791 |
| Pneumonia | H2 | 2.693761 | 0.050556 | 68 | 791 |
| Pneumonia | H3 | 2.728186 | 0.057038 | 68 | 791 |
| Edema | H0 | 1.053384 | 0.156241 | 115 | 744 |
| Edema | H1 | 0.777364 | 0.260709 | 115 | 744 |
| Edema | H2 | 1.039401 | 0.159720 | 115 | 744 |
| Edema | H3 | 1.455895 | 0.102722 | 115 | 744 |
| Cardiomegaly | H0 | 1.094186 | 0.247759 | 177 | 682 |
| Cardiomegaly | H1 | 1.063507 | 0.298176 | 177 | 682 |
| Cardiomegaly | H2 | 1.043283 | 0.259996 | 177 | 682 |
| Cardiomegaly | H3 | 1.215414 | 0.243096 | 177 | 682 |
| Atelectasis | H0 | 1.183735 | 0.217106 | 193 | 666 |
| Atelectasis | H1 | 1.055656 | 0.273966 | 193 | 666 |
| Atelectasis | H2 | 1.116494 | 0.235958 | 193 | 666 |
| Atelectasis | H3 | 1.055711 | 0.303030 | 193 | 666 |
| Lung Opacity | H0 | 1.310825 | 0.235839 | 214 | 645 |
| Lung Opacity | H1 | 1.325949 | 0.242299 | 214 | 645 |
| Lung Opacity | H2 | 1.369211 | 0.219275 | 214 | 645 |
| Lung Opacity | H3 | 1.206483 | 0.283892 | 214 | 645 |
| Pleural Effusion | H0 | 0.685289 | 0.215468 | 260 | 599 |
| Pleural Effusion | H1 | 0.758269 | 0.202228 | 260 | 599 |
| Pleural Effusion | H2 | 0.605639 | 0.246786 | 260 | 599 |
| Pleural Effusion | H3 | 0.619298 | 0.331897 | 260 | 599 |
| No Finding | H0 | 0.688984 | 0.265568 | 306 | 553 |
| No Finding | H1 | 0.855354 | 0.223859 | 306 | 553 |
| No Finding | H2 | 0.728522 | 0.249720 | 306 | 553 |
| No Finding | H3 | 0.790710 | 0.263771 | 306 | 553 |

> 注意：仅凭某类总体 BCE 高低不能判断疾病难度，因为阳性率不同；positive/negative 拆分与 AUROC 需一起看。

## D. AUROC / AUPRC 变化（Best Checkpoint）

| Disease | H0 Valid AUROC | H1 Valid AUROC | H2 Valid AUROC | H3 Valid AUROC | H0 Valid AUPRC | H1 Valid AUPRC | H2 Valid AUPRC | H3 Valid AUPRC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Enlarged Cardiomediastinum | 0.707859 | 0.714421 | 0.717440 | 0.701173 | 0.097229 | 0.075395 | 0.090365 | 0.055632 |
| Pneumothorax | 0.855248 | 0.793976 | 0.856530 | 0.768199 | 0.351232 | 0.139873 | 0.306424 | 0.162747 |
| Consolidation | 0.788465 | 0.752433 | 0.793732 | 0.741109 | 0.199799 | 0.150916 | 0.197173 | 0.178024 |
| Pneumonia | 0.724195 | 0.696466 | 0.718181 | 0.664275 | 0.203233 | 0.174763 | 0.195093 | 0.139152 |
| Edema | 0.885361 | 0.867584 | 0.885799 | 0.869086 | 0.478239 | 0.451919 | 0.479321 | 0.410472 |
| Cardiomegaly | 0.776882 | 0.763412 | 0.782792 | 0.739425 | 0.430919 | 0.401395 | 0.439611 | 0.390301 |
| Atelectasis | 0.792594 | 0.772390 | 0.792618 | 0.729963 | 0.462141 | 0.454736 | 0.462046 | 0.407710 |
| Lung Opacity | 0.723332 | 0.727019 | 0.725049 | 0.702550 | 0.457379 | 0.444967 | 0.467758 | 0.452680 |
| Pleural Effusion | 0.908774 | 0.901416 | 0.909574 | 0.869902 | 0.776823 | 0.775842 | 0.784624 | 0.731696 |
| No Finding | 0.871979 | 0.869228 | 0.868666 | 0.852176 | 0.827054 | 0.816395 | 0.827164 | 0.797958 |

## E. 机制分析

### E1. H1 AP-only 采样

- 实际 AP-only 采样比例：均值 0.5500，最小 0.5500，最大 0.5500（目标 0.55）。
- 每 Epoch 抽样 Study 次数：18513；唯一 Study 数均值 11665.9；重复≥3 次 Study 数均值 1510.0；从未被抽到的 Study 数均值 6847.1。
- Subject 重复：平均每 Epoch 被抽 2.34 次，最大 31 次。
- AP-only 与 PA+Lateral 的 AUROC/BCE 对比（Best，Validation）见 `grouped_performance_comparison.csv` 与下表。

| Experiment | Group | Macro-AUROC | Macro-AUPRC | Overall BCE |
|---|---|---:|---:|---:|
| H0 | AP-only | 0.724279 | 0.403985 | 0.157224 |
| H0 | PA+Lateral | 0.868411 | 0.465589 | 0.079337 |
| H0 | Single-image | 0.727877 | 0.402549 | 0.171979 |
| H0 | Multi-image | 0.849957 | 0.447634 | 0.085522 |
| H1 | AP-only | 0.698154 | 0.356307 | 0.152951 |
| H1 | PA+Lateral | 0.851385 | 0.416479 | 0.072112 |
| H1 | Single-image | 0.708565 | 0.364660 | 0.167017 |
| H1 | Multi-image | 0.830374 | 0.396626 | 0.078727 |
| H2 | AP-only | 0.726325 | 0.396567 | 0.152479 |
| H2 | PA+Lateral | 0.870406 | 0.465177 | 0.073833 |
| H2 | Single-image | 0.729328 | 0.394330 | 0.168233 |
| H2 | Multi-image | 0.851752 | 0.452623 | 0.079901 |
| H3 | AP-only | 0.684753 | 0.347450 | 0.149669 |
| H3 | PA+Lateral | 0.825530 | 0.391041 | 0.081607 |
| H3 | Single-image | 0.694360 | 0.361311 | 0.161945 |
| H3 | Multi-image | 0.804185 | 0.362657 | 0.088618 |

### E2. H2 单图 Study 损失权重

- 单图 Study 训练样本数均值：8507.0；多图 Study 均值：10006.0。
- 单图原始 BCE 均值：0.328834；多图原始 BCE 均值：0.218216。
- 归一化加权目标 BCE 均值：0.275207；同 Epoch 未加权 BCE 均值：0.269046。
- 单图/多图 AUROC、BCE 对比见 `grouped_performance_comparison.csv`；H2 的加权目标仅用于训练，所有比较使用原始未加权 BCE。

### E3. H3 L2-SP

- λ_SP（Train-only 梯度尺度校准，目标 L2-SP 梯度范数 ≈ BCE 梯度范数 5%）：**2.89278e+07**。
- 校准时 BCE 梯度范数 21.7768，单位 λ L2-SP 梯度范数 3.76399e-08；校准后 Stage3/4 相对漂移 0.001229 / 0.001175。
- 每 Epoch L2-SP 项均值：6.445157e-10；λ·R_SP 均值：0.018644。
- Stage3 相对漂移（逐 Epoch 末）：0.000321；Stage4：0.000227。
- BCE/L2-SP 梯度范数比（λ·‖g_SP‖/‖g_BCE‖，逐 Epoch 起始）：均值 0.0095；该比例随参数偏离增大而增长，是 L2-SP 的固有性质。

| Experiment | Stage3 Best Relative Drift | Stage4 Best Relative Drift |
|---|---:|---:|
| H0 | 0.008271 | 0.007639 |
| H1 | 0.006358 | 0.006086 |
| H2 | 0.008268 | 0.007638 |
| H3 | 0.000321 | 0.000227 |

- 需要明确：Validation 表现下降不能证明 Test 一定下降，但也不能证明 Test 会提高；H3 的结论限定在当前验证协议与单 seed 条件下。

## F. 综合判断

### F1. 三个假设的证据状态

| Hypothesis | Valid Macro-AUROC Δ vs H0 | Valid BCE Δ vs H0 | 是否支持 |
|---|---:|---:|---|
| H1 | -0.017634 | +0.018779 | CI 排除 0 且为负 |
| H2 | +0.001569 | -0.000088 | CI 跨 0，无法确认 |
| H3 | -0.039683 | +0.020979 | CI 排除 0 且为负 |

### F2. 关键现象检查

- H1: Train BCE Δ=+0.030181，Valid BCE Δ=+0.018779，Valid Macro-AUROC Δ=-0.017634。
- H2: Train BCE Δ=-0.001149，Valid BCE Δ=-0.000088，Valid Macro-AUROC Δ=+0.001569。
- H3: Train BCE Δ=+0.079324，Valid BCE Δ=+0.020979，Valid Macro-AUROC Δ=-0.039683。

### F3. 统计不确定性与结论边界

- H1−H0: Δ Macro-AUROC = -0.017634, 95% CI [-0.028003239631652833, -0.006522588431835181]。
- H2−H0: Δ Macro-AUROC = +0.001569, 95% CI [-0.0020014196634292603, 0.005501404404640188]。
- H3−H0: Δ Macro-AUROC = -0.039683, 95% CI [-0.05647059231996536, -0.022576835751533533]。
- 这些区间是同一 Validation 上的 Subject-cluster 配对 bootstrap，属于验证集被重复使用条件下的探索性证据；单 seed 的万分位差异不足以宣布有效。
- 官方 Test 标签未读取，无法把任何 Validation 改善解释为 Test 收益；下一步若扩展 seed（2026/3407）或组合策略，需要新的确认性实验。


## G. 验收问题回答（12 项）

1. **H1 是否改善 AP-only Study 的识别能力？** 否。AP-only 子群 Macro-AUROC：H0 0.724279 → H1 0.698154（Δ=-0.026124），且 H1 总体验证 AUROC 显著下降（Δ=-0.017634，95% CI [-0.028003239631652833, -0.006522588431835181]）。
2. **H2 是否改善单图 Study 的识别能力？** 方向为正但不确定：单图子群 Macro-AUROC H0 0.727877 → H2 0.729328（Δ=+0.001451）；多图子群 H0 0.849957 → H2 0.851752（Δ=+0.001795）；总体 Δ=+0.001569，95% CI [-0.0020014196634292603, 0.005501404404640188] 跨 0。
3. **H3 是否降低 Stage3/4 参数漂移？** 是。Best checkpoint 的 Stage3 相对漂移 H0 0.008271 → H3 0.000321；Stage4 H0 0.007639 → H3 0.000227（降低约 20 倍）。
4. **H3 是否改善 Validation BCE 或 Macro-AUROC？** 在本轮预算内没有：Valid Overall BCE H0 0.311185 → H3 0.332165（变差）；Valid Macro-AUROC 0.803469 → 0.763786（Δ=-0.039683，95% CI [-0.05647059231996536, -0.022576835751533533]）。但 H3 在 epoch 12 停止时仍缓慢上升，三条件早停未触发，因此这不是收敛结论。
5. **H1/H2/H3 对十类疾病 Train Loss 的影响？** 相对 H0 的 Train BCE 增大最多：H1 No Finding +0.061987；Edema +0.061702；Pleural Effusion +0.044013；Cardiomegaly +0.037663；H2 Pneumonia +0.001545；Cardiomegaly +0.000280；Pleural Effusion +0.000278；No Finding -0.000199；H3 No Finding +0.165795；Pleural Effusion +0.146019；Atelectasis +0.097347；Cardiomegaly +0.096688。
6. **对十类疾病 Valid Loss 的影响？** 相对 H0 的 Valid BCE 增大最多：H1 Edema +0.053529；Cardiomegaly +0.033707；No Finding +0.032414；Pneumothorax +0.017982；H2 No Finding +0.003882；Pneumonia +0.003225；Lung Opacity +0.002108；Edema +0.001141；H3 Pleural Effusion +0.061215；Atelectasis +0.037854；No Finding +0.035081；Cardiomegaly +0.021277。具体逐类数值见上表 B。
7. **哪些疾病的 Train–Valid Loss Gap 最大？** No Finding(H2) +0.159751；No Finding(H0) +0.155670；Pleural Effusion(H0) +0.137250；Pleural Effusion(H2) +0.134702；No Finding(H1) +0.126097。总体 Gap 最大为 H0 +0.075051（H2 +0.076112，H1 +0.063649，H3 +0.016706）。
8 (H1). **哪些疾病的 Valid Loss 得到改善（H1 vs H0）？** Enlarged Cardiomediastinum -0.005961。改善类别数：1/10。
8 (H2). **哪些疾病的 Valid Loss 得到改善（H2 vs H0）？** Enlarged Cardiomediastinum -0.004767；Pneumothorax -0.003047；Pleural Effusion -0.002270；Cardiomegaly -0.000774；Atelectasis -0.000492。改善类别数：5/10。
8 (H3). **哪些疾病的 Valid Loss 得到改善（H3 vs H0）？** Enlarged Cardiomediastinum -0.002881。改善类别数：1/10。
9. **是否存在 Train Loss 下降但 Valid Loss 上升的现象？** 三个实验的总体 Train/Valid 方向均不完全满足该模式；逐疾病层面存在局部Train 下降/Valid 上升（见 `per_class_loss_gap.csv`）。
10. **是否存在 Valid Loss 改善但 AUROC 不变甚至下降的现象？** 逐疾病统计（Valid BCE 改善且 AUROC 不升的类别数）：H1 0/10、H2 0/10、H3 1/10。该现象说明 BCE 改善可能来自概率校准/尺度变化，而不一定是排序能力提升。
11. **哪种方法表现出最值得进一步验证的泛化收益？** H2（单图 Study 损失权重）是唯一总体方向为正的实验：Valid Macro-AUROC 0.803469 → 0.805038（Δ=+0.001569），单图子群 Δ=+0.001451，但要允许 95% CI [-0.0020014196634292603, 0.005501404404640188] 跨 0 与 AUPRC 轻微下降（0.428405 → 0.424958）。H1 为明确负向；H3 为未收敛的负向。
12. **下一步应增加配对随机种子，还是研究策略组合？** 现阶段应先对 H2 做配对随机种子确认（seed 2026/3407，固定同一矩阵），而不是直接做组合；H1 不建议扩展，H3 需要先修正收敛/预算问题（或重新校准 L2-SP 强度）后再谈组合。

