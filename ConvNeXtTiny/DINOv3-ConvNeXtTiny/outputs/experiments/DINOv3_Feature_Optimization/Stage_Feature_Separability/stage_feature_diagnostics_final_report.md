# R5 / R6 四阶段疾病特征可分性诊断

## 摘要

- R5 Best：Epoch 6，Macro-AUROC 0.803514；R6 Best：Epoch 6，Macro-AUROC 0.798802。
- 正式模型 R6−R5 Macro-AUROC：-0.004712，95% CI [-0.014618, +0.002907]；Macro-AUPRC 差值 +0.003663，95% CI [-0.018225, +0.019995]。
- 下表展示冻结 Study 特征经 Subject 分组 Train-CV 选择 C 的 L2 Logistic Regression Probe；不是 R5/R6 正式模型分数。

## 1. 正式 R6 对比 R5

完整逐类监督模型指标与训练状态见 `R6_Full_Finetune_512/r5_vs_r6_comparison.md`。全量解冻后是否改善必须结合配对 Subject Bootstrap 区间；Validation 已用于 Best checkpoint 选择，统计结果是探索性证据。

| Run | Best Epoch | Macro-AUROC | Train BCE at Best | Valid BCE at Best | Optimizer Updates through Best | Mean images/s through Best | Peak reserved GiB | Trainable Params |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R5 | 6 | 0.803514 | 0.262511 | 0.311168 | 6,037 | 22.14 | 1.57 | 26,592,778 |
| R6 | 6 | 0.798802 | 0.265861 | 0.314805 | 6,037 | 22.08 | 4.04 | 27,827,818 |

## 2. R5 / R6 Stage × Disease AUROC Probe 矩阵

AUROC 与 AUPRC 矩阵及对应 Bootstrap CI 位于 `stage_auroc_matrix_*.csv`、`stage_auprc_matrix_*.csv` 和 `stage_paired_bootstrap.json`。

| Disease | R5 Best Stage (AUROC) | R6 Best Stage (AUROC) | R5 Stage3→4 Δ | R6 Stage3→4 Δ | R5 Stage3+4 vs Stage4 | R6 Stage3+4 vs Stage4 |
|---|---|---|---:|---:|---:|---:|
| Enlarged Cardiomediastinum | Stage 4 (0.731) | Stage 3 (0.718) | 0.0203 [-0.0764, 0.1302] | -0.0088 [-0.0871, 0.0661] | -0.0359 [-0.0762, -0.0081] | 0.0083 [-0.0048, 0.0234] |
| Pneumothorax | Stage 3 (0.825) | Stage 4 (0.822) | -0.0200 [-0.0975, 0.0351] | 0.0042 [-0.0498, 0.0496] | -0.0139 [-0.0406, 0.0046] | -0.0166 [-0.0365, -0.0006] |
| Consolidation | Stage 4 (0.781) | Stage 4 (0.753) | 0.0586 [0.0044, 0.1224] | 0.0297 [-0.0248, 0.0789] | 0.0138 [-0.0063, 0.0343] | 0.0268 [0.0094, 0.0477] |
| Pneumonia | Stage 3 (0.678) | Stage 4 (0.692) | -0.0057 [-0.0712, 0.0549] | 0.0246 [-0.0286, 0.0795] | 0.0269 [0.0042, 0.0532] | -0.0004 [-0.0206, 0.0177] |
| Edema | Stage 4 (0.879) | Stage 3 (0.866) | 0.0146 [-0.0024, 0.0311] | -0.0063 [-0.0294, 0.0146] | -0.0012 [-0.0046, 0.0024] | 0.0066 [-0.0009, 0.0149] |
| Cardiomegaly | Stage 4 (0.765) | Stage 4 (0.765) | 0.0074 [-0.0219, 0.0337] | 0.0013 [-0.0263, 0.0254] | 0.0040 [-0.0052, 0.0145] | 0.0042 [-0.0040, 0.0114] |
| Atelectasis | Stage 4 (0.772) | Stage 4 (0.778) | 0.0051 [-0.0176, 0.0282] | 0.0161 [-0.0085, 0.0415] | 0.0094 [0.0024, 0.0165] | 0.0050 [-0.0030, 0.0128] |
| Lung Opacity | Stage 3 (0.736) | Stage 4 (0.727) | -0.0148 [-0.0508, 0.0156] | 0.0047 [-0.0238, 0.0342] | 0.0074 [-0.0009, 0.0177] | 0.0027 [-0.0063, 0.0114] |
| Pleural Effusion | Stage 4 (0.906) | Stage 4 (0.906) | 0.0058 [-0.0050, 0.0166] | 0.0076 [-0.0038, 0.0177] | 0.0021 [-0.0013, 0.0054] | 0.0010 [-0.0005, 0.0027] |
| No Finding | Stage 3 (0.856) | Stage 3 (0.850) | -0.0088 [-0.0266, 0.0086] | -0.0205 [-0.0382, -0.0043] | -0.0020 [-0.0068, 0.0022] | -0.0028 [-0.0086, 0.0033] |

## 3. 弱类别与对照类别

### Pneumonia
- R5: AUROC Stage1→4 `0.585, 0.589, 0.678, 0.673`；AUPRC `0.104, 0.102, 0.173, 0.168`。
- Stage4−Stage3 AUROC Δ -0.0057 [-0.0712, +0.0549]；Stage3+4−Stage4 Δ +0.0269 [+0.0042, +0.0532].
- R6: AUROC Stage1→4 `0.596, 0.640, 0.668, 0.692`；AUPRC `0.099, 0.132, 0.171, 0.173`。
- Stage4−Stage3 AUROC Δ +0.0246 [-0.0286, +0.0795]；Stage3+4−Stage4 Δ -0.0004 [-0.0206, +0.0177].

### Lung Opacity
- R5: AUROC Stage1→4 `0.649, 0.648, 0.736, 0.721`；AUPRC `0.376, 0.365, 0.455, 0.439`。
- Stage4−Stage3 AUROC Δ -0.0148 [-0.0508, +0.0156]；Stage3+4−Stage4 Δ +0.0074 [-0.0009, +0.0177].
- R6: AUROC Stage1→4 `0.652, 0.680, 0.722, 0.727`；AUPRC `0.360, 0.401, 0.453, 0.440`。
- Stage4−Stage3 AUROC Δ +0.0047 [-0.0238, +0.0342]；Stage3+4−Stage4 Δ +0.0027 [-0.0063, +0.0114].

### Consolidation
- R5: AUROC Stage1→4 `0.658, 0.689, 0.722, 0.781`；AUPRC `0.082, 0.088, 0.144, 0.159`。
- Stage4−Stage3 AUROC Δ +0.0586 [+0.0044, +0.1224]；Stage3+4−Stage4 Δ +0.0138 [-0.0063, +0.0343].
- R6: AUROC Stage1→4 `0.670, 0.686, 0.723, 0.753`；AUPRC `0.084, 0.098, 0.149, 0.175`。
- Stage4−Stage3 AUROC Δ +0.0297 [-0.0248, +0.0789]；Stage3+4−Stage4 Δ +0.0268 [+0.0094, +0.0477].

### Pneumothorax
- R5: AUROC Stage1→4 `0.747, 0.751, 0.825, 0.805`；AUPRC `0.104, 0.162, 0.187, 0.226`。
- Stage4−Stage3 AUROC Δ -0.0200 [-0.0975, +0.0351]；Stage3+4−Stage4 Δ -0.0139 [-0.0406, +0.0046].
- R6: AUROC Stage1→4 `0.771, 0.767, 0.818, 0.822`；AUPRC `0.127, 0.218, 0.253, 0.363`。
- Stage4−Stage3 AUROC Δ +0.0042 [-0.0498, +0.0496]；Stage3+4−Stage4 Δ -0.0166 [-0.0365, -0.0006].

## 4. 全量微调改变了哪些阶段？

逐疾病、逐 Stage 的 R6−R5 AUROC/AUPRC 及 paired CI 见 `r6_minus_r5_stage_deltas.csv` 与对应热力图。R6 与 R5 特征空间不同；Probe 变化描述可分性差异，不将通道数值对应解释为相同医学语义。

## 5. Stage 互补性与特征维度敏感性

`stage_joint_probe_comparison.csv` 对 Stage3+4 与两者单独 Probe，以及 Stage2+3 与 Stage2/3 单独 Probe 进行同 Study 配对比较。只有配对 CI 不跨零时，才将验证结果描述为当前 Probe 协议下的可检测补充价值；这不证明 Stage4 不可逆地丢失病理信息。`pca64_probe_comparison.csv` 比较各 Stage 原维度和统一 PCA-64 后的 Probe。

## 6. 训练曲线

- R5 实际训练 9 Epoch，Best Epoch 6；Train BCE 0.1425、Validation BCE 0.4535。
- R6 实际训练 9 Epoch，Best Epoch 6；Train BCE 0.1478、Validation BCE 0.4447。
训练曲线和停止规则见 `charts/` 及各自训练目录。

## 7. 下一轮研究判断

- 四个单 Stage Probe 的 Macro-AUROC：R5 Stage1/2/3/4=`0.7116/0.7215/0.7818/0.7880`；R6=`0.7189/0.7354/0.7789/0.7842`。当前证据不支持把主要瓶颈概括为 Stage3 特征获取不足；判别性主要在 Stage3/4 已形成。
- Stage3+Stage4 相对单独 Stage4 的 Macro-AUROC 增量：R5 `+0.0011`，95% CI `[-0.0058,+0.0067]`；R6 `+0.0035`，95% CI `[-0.0005,+0.0073]`。宏平均没有清楚的联合收益。
- 存在类别级互补信号：R6 Consolidation 的 Stage3+4−Stage4 为 `+0.0268`，95% CI `[+0.0094,+0.0477]`，且 Stage3+4−Stage3 也为正；R5 Pneumonia 的 Stage3+4−Stage4 为 `+0.0269`，95% CI `[+0.0042,+0.0532]`，但相对 Stage3 的区间跨零。因为比较数量多且 Validation 已参与历史选模，这些结果是需要独立复验的候选假设。
- 因此若设计下一轮结构验证，优先测试一个轻量 Stage3→Stage4/Global GAP 特征融合，预先指定 Consolidation 为主要类别、Pneumonia 为次要类别，同时保留普通 R5 GAP 对照；本轮不实现也不启动新训练。
- R6 正式 Macro-AUROC 点估计低于 R5 `0.0047`，配对 95% CI `[-0.0146,+0.0029]` 跨零，且 R6 可训练参数更多。当前保留 R5 Global-only 作为后续结构改进基线，R6 保留为全量微调对照；这不构成 R6 明确劣于 R5 的统计证据。

## 8. 统计与方法限制

- Validation：859 Studies / 228 Subjects；标签为官方 closed-world 十类。R5 与 R6 使用相同的 Validation Study keys 和标签顺序。Bootstrap 以 Subject 为单位，Seed=42，2000 次；同一 Study 的多张图在特征提取阶段先做 image-level GAP，再按 Study 平均。
- 正则化强度只在 Train 内部 Subject 分组 3-fold CV（若折内类别缺失则降为 2-fold）选择；Scaler、PCA 只在当前 CV 训练折拟合。Train OOF 概率选择补充 F1 阈值；正式 Validation 未用于阈值或 C 选择。
- 对重复疾病、Stage 与 Probe 比较存在多重比较风险；Validation 还曾用于历史 checkpoint 选择。Bootstrap CI 是探索性不确定性描述，不替代独立测试集确认。
- Probe 测量线性可分性与当前预处理/聚合下的特征可用性，不证明因果机制、真实病灶定位或临床有效性。未使用 Test。
