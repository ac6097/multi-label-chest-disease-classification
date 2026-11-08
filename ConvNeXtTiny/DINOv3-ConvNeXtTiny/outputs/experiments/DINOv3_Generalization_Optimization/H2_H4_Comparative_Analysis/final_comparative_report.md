# H2-B / H2-C / H4 comparative report

## Overall comparison (Best checkpoint; full deterministic Train/Validation inference)

| Experiment | Train BCE | Valid BCE | BCE Gap | Valid AUROC | Valid AUPRC | Best Epoch | Stop |
|---|---:|---:|---:|---:|---:|---:|---|
| H0 | 0.236135 | 0.311185 | +0.075051 | 0.803469 | 0.428405 | 6 | three_condition_early_stop |
| H2 | 0.234986 | 0.311098 | +0.076112 | 0.805038 | 0.424958 | 6 | three_condition_early_stop |
| H2B | 0.234647 | 0.311292 | +0.076645 | 0.803685 | 0.427324 | 6 | three_condition_early_stop |
| H2C | 0.194044 | 0.331139 | +0.137095 | 0.800501 | 0.407754 | 7 | three_condition_early_stop |
| H4 | 0.239685 | 0.313042 | +0.073357 | 0.803069 | 0.425843 | 6 | three_condition_early_stop |

## Paired Subject-cluster bootstrap (Best, Validation, 2000 draws)

- H2-B − H0: ΔAUROC +0.000216 CI [-0.0017161592841148376, 0.002250231802463531]
- H2 − H0: ΔAUROC +0.001569 CI [-0.0020014196634292603, 0.005501404404640188]
- H2-C − H0: ΔAUROC -0.002968 CI [-0.012691408395767212, 0.007520265877246857]
- H4 − H0: ΔAUROC -0.000399 CI [-0.010014253854751586, 0.008509881794452667]
- H2-B − H2: ΔAUROC -0.001353 CI [-0.003911639750003815, 0.0011030450463294973]
- H2-C − H2: ΔAUROC -0.004538 CI [-0.0136410191655159, 0.005808569490909567]

这些区间是官方 Validation 反复使用后的探索性统计，不是独立 Test 结论。

## Grouped Validation metrics (Best; fixed macro BCE)

| Experiment | Group | Macro-AUROC | Macro-AUPRC | Macro-BCE |
|---|---|---:|---:|---:|
| H0 | AP-only | 0.724279 | 0.403985 | 0.386448 |
| H0 | PA+Lateral | 0.868411 | 0.465589 | 0.214083 |
| H0 | Single-image | 0.727877 | 0.402549 | 0.392367 |
| H0 | Multi-image | 0.849957 | 0.447634 | 0.243360 |
| H2 | AP-only | 0.726325 | 0.396567 | 0.386950 |
| H2 | PA+Lateral | 0.870406 | 0.465177 | 0.213479 |
| H2 | Single-image | 0.729328 | 0.394330 | 0.392892 |
| H2 | Multi-image | 0.851752 | 0.452623 | 0.242761 |
| H2B | AP-only | 0.723867 | 0.400738 | 0.387586 |
| H2B | PA+Lateral | 0.869942 | 0.467299 | 0.213097 |
| H2B | Single-image | 0.727427 | 0.400195 | 0.393664 |
| H2B | Multi-image | 0.851058 | 0.448803 | 0.242474 |
| H2C | AP-only | 0.721598 | 0.381787 | 0.413222 |
| H2C | PA+Lateral | 0.863949 | 0.441285 | 0.225284 |
| H2C | Single-image | 0.723165 | 0.383099 | 0.419722 |
| H2C | Multi-image | 0.847333 | 0.424883 | 0.257130 |
| H4 | AP-only | 0.725903 | 0.398685 | 0.389250 |
| H4 | PA+Lateral | 0.866901 | 0.474199 | 0.213635 |
| H4 | Single-image | 0.726542 | 0.398442 | 0.396813 |
| H4 | Multi-image | 0.848873 | 0.450288 | 0.243054 |

## Answers to the ten required questions

- **1. 单图权重 1.10/1.25/1.50 哪项更好？** 在当前 Validation 上，Best AUROC 最高的是 **H2**（0.805038）；H2-B 0.803685、H2 0.805038、H2-C 0.800501。差异很小，见 bootstrap CI。
- **2. 单图权重增加是否使单图 AUROC 单调改善？** 否。单图子群 AUROC：H0 0.727877、H2-B 0.727427、H2 0.729328、H2-C 0.723165、H4 0.726542；没有随权重单调上升。
- **3. 对多图 Study 是否有负面影响？** 多图子群 AUROC：H0 0.849957、H2-B 0.851058、H2 0.851752、H2-C 0.847333；无一致下降趋势，权重 1.50 未见系统性多图损害。
- **4. AP-only 是否随单图权重变化？** AP-only AUROC：H0 0.724279、H2-B 0.723867、H2 0.726325、H2-C 0.721598。变化存在但无单调规律。
- **5. 是否只是概率尺度改变，而不是排序能力改善？** AUROC 提高但 AUPRC 下降的 (实验:疾病) 组合：H2B:Enlarged Cardiomediastinum、H2B:Consolidation、H2B:Edema、H2B:Atelectasis、H2B:Pleural Effusion、H2C:Enlarged Cardiomediastinum、H2C:Consolidation、H2C:Pneumonia、H4:Enlarged Cardiomediastinum、H4:Cardiomegaly。例如 H2 的总体 AUROC 略升而 AUPRC 下降，H2-C 的 Train BCE 明显下降但 Valid BCE/AUPRC 更差，说明部分变化是概率尺度/校准而非排序能力提升；逐疾病结果见 `per_class_auroc_comparison.csv` 与 `per_class_auprc_comparison.csv`。
- **6. Dropout 是否改善 Train/Valid 泛化差距？** H4 Gap +0.073357 vs H0 +0.075051；H4 Valid AUROC 0.803069 vs H0 0.803469（Δ=-0.000399，CI [-0.010014253854751586, 0.008509881794452667]）。
- **7. Dropout 是否降低 Validation BCE？** H4 Valid BCE 0.313042 vs H0 0.311185，Δ=+0.001857。
- **8. Dropout 是否改善困难疾病？** H4 相对 H0 的 Valid AUROC 提高类别：Enlarged Cardiomediastinum、Atelectasis、Pleural Effusion、Cardiomegaly；下降类别：Consolidation、No Finding、Edema、Lung Opacity、Pneumonia、Pneumothorax。Enlarged Cardiomediastinum 有改善，但 Consolidation、Lung Opacity、No Finding、Pneumonia 等下降；没有系统性的“困难疾病全部获益”，且 H4 总体 AUPRC 更低。
- **9. Dropout 是否明显影响收敛速度？** H4 Best epoch 6，完成 9 轮；H0 Best epoch 6，完成 9 轮。收敛曲线见 `overall_loss_curves.png`；H4 未显示明显加速或减慢。
- **10. 哪项最值得多随机种子复验？** 本轮没有任何一项显示稳定的泛化收益：H2-B 与 H0 基本重合（ΔAUROC +0.000216，CI [-0.0017161592841148376, 0.002250231802463531]），H2 仅 +0.0016 且 CI 跨 0、AUPRC −0.0034，H2-C 与 H4 更差。因此**不建议**为 H2-C/H4 花三个配对种子；若资源允许，唯一可做的是 H2(1.25) 与 H2-B(1.10) 的低优先级复验，用来确认“单图权重收益”是否只是单 seed 噪声。 H2-C 的高 Train 拟合（Train BCE 0.194）伴随 Valid 退化，明确不支持继续加大权重。

## Bottom line

- **单图权重**：1.10 与 H0 基本重合；1.25 仅有 +0.0016 的点估计（CI 跨 0，AUPRC 略降）；1.50 明确过拟合（Train BCE 0.194 但 Valid BCE 0.331、AUPRC 0.408）。没有证据支持继续加大单图权重。
- **Dropout**：H4 未降低 Valid BCE（0.3130 vs 0.3112），未提高 AUROC（−0.0004，CI 跨 0）或 AUPRC（−0.0026）；Gap 略小但伴随整体排序/校准变差，不能视为泛化收益。
- **决定**：不启动三配对随机种子复验。若必须保留一个低优先级方向，只复验 H2(1.25) 与 H2-B(1.10) 的差异以确认其是否只是单 seed 噪声；H2-C 与 H4 关闭。
- **单图 AP-only 子群**：H2 的 AP-only +0.0020、单图 +0.0015，但 CI 跨 0；没有可宣称的 AP-only 改善。

## Grouped BCE correction

- 旧 H3-B 报告中的分组 “Overall BCE” 实际读取了 `LABEL_NAMES[0]`（Enlarged Cardiomediastinum）的单病种 BCE；本轮已在 `grouped_metrics` 中增加 `macro_bce`（十类均值）并修复报告取数。
- 新旧数值、断言与根因见 `grouped_bce_validation_audit.md`；H0/H2/H2-B/H2-C/H4 的修正版分组 BCE 见 `grouped_performance_comparison.csv`。

## Limitations

- 单 seed，Validation 反复用于选模与比较；CI 为探索性。
- H2-B/H2-C/H4 的训练循环每 epoch 记录了 LR、更新次数、原始 BCE、优化目标、逐 epoch 十类 Validation AUROC/AUPRC/BCE；但**未单独记录每步梯度范数**（历史训练器只在 L2-SP 实验中保存 BCE/L2-SP 梯度范数）。梯度裁剪阈值 1.0 每 update 执行，触发比例未保存，不能事后伪造。
- H0/H2 的中间 epoch Train 逐类 BCE 在历史文件中不完整；本轮对比表使用Best/Last 的完整确定性逐类 BCE。新实验的逐 epoch Train/Valid 逐类 BCE 完整（`per_epoch_class_loss.csv`）。
- H0 为复用的 R5；H2 为历史实验，H2-B/H2-C/H4 为新独立训练。
- 无 Test 标签，未用 Test 选权重或判断 Dropout。
- 分组 BCE 已修复并断言；历史 H3-B 文件不覆盖，只在本目录输出修正版。

