# H0 / H3 / H3-B comparative report

| Experiment | Best Epoch | Stop | Train BCE | Valid BCE | BCE Gap | Valid Macro-AUROC | Valid Macro-AUPRC | Stage3 drift | Stage4 drift |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| H0 | 6 | three_condition_early_stop | 0.236135 | 0.311185 | +0.075051 | 0.803469 | 0.428405 | 0.008271 | 0.007639 |
| H3 | 12 | resource_budget_stop | 0.315458 | 0.332165 | +0.016706 | 0.763786 | 0.372637 | 0.000321 | 0.000227 |
| H3B | 13 | three_condition_early_stop | 0.290968 | 0.316011 | +0.025044 | 0.791497 | 0.413933 | 0.001139 | 0.000664 |

## Convergence speed (training-batch raw BCE / full-inference Best AUROC)

| Experiment | Best epoch | Completed epochs | Epochs to Train BCE ≤0.35 | ≤0.30 | ≤0.28 | ≤0.26 | Best Valid AUROC |
|---|---:|---:|---:|---:|---:|---:|---:|
| H0 | 6 | 9 | 2 | 4 | 6 | 7 | 0.803469 |
| H3 | 12 | 12 | 4 | — | — | — | 0.763786 |
| H3B | 13 | 19 | 2 | 13 | 16 | 18 | 0.791497 |

- H0 在 epoch 6 达到 Best；H3 在 epoch 12（预算停止）；H3-B 在 epoch 13 达到 Best，并在 epoch 19 触发三条件早停。
- 收敛速度：H3-B 仍慢于 H0（Train BCE 到 0.30 所需 epoch 见上表），但明显快于 H3；H3-B 的 Last checkpoint 已开始过拟合（见下）。

## Answers to the ten required questions

1. **H3 的 L2-SP 实现是否正确？** 见 `l2sp_implementation_audit.md`：参数范围、预训练对应关系、element-mean 归一化、λ 比例、梯度公式、梯度累积单次反传、分类头/冻结排除、AdamW weight decay 与 LR 全部通过；未发现重复 λ 或 microbatch 放大。
2. **新 λ 是否确实降低了正则强度？** 是。H3 λ=2.89278e+07，H3-B λ=2.89278e+06，精确 0.1 倍；同漂移点下 L2-SP 梯度范数从 H3 的 1.089 降至约 0.1 倍；Stage3/4 漂移从 H3 的 0.000321/0.000227 上升（见上表），说明约束确实减弱。
3. **H3-B 是否恢复了 Train BCE / Train AUROC？** Train BCE H0 0.236135、H3 0.315458、H3-B 0.290968；H3-B 明显低于 H3，但仍高于 H0；Train Macro-AUROC：H0 0.903096、H3 0.793639、H3-B 0.837969。
4. **H3-B 是否正常收敛 / 早停？** `three_condition_early_stop`，完成 19/20 epochs，Best epoch 13；最后三轮数据见 `h3b_final_report.md`。
5. **Validation AUROC 是否恢复到接近 H0？** H3-B 0.791497 vs H0 0.803469（Δ=-0.011971，95% CI [-0.025593961775302886, 0.0011581733822822497]），较 H3 的 −0.0397 明显恢复，但仍未达到 H0 点估计。
6. **哪些疾病从降低正则化强度中获益最大？** 按 Valid BCE 相对 H3 改善排序：Pleural Effusion -0.054678；No Finding -0.025131；Atelectasis -0.022782；Cardiomegaly -0.017275；Pneumothorax -0.015568（负值表示 H3-B 更低，见 `per_class_benefit_vs_h3.csv`）。
7. **Stage3/4 参数漂移如何变化？** Stage3：H0 0.008271、H3 0.000321、H3-B 0.001139；Stage4 对应为 0.007639 / 0.000227 / 0.000664。
8. **Train/Valid BCE Gap 如何变化？** H0 +0.075051、H3 +0.016706、H3-B +0.025044；H3-B 的 Gap 仍小于 H0，因为 Train BCE 仍偏高（欠拟合），不能把 Gap 缩小直接当成泛化改善。
9. **单图与 AP-only 子群是否改善？** AP-only：H0 0.724279、H3 0.684753、H3-B 0.712055（H3-B 比 H3 +0.027302，比 H0 -0.012224）；单图：H0 0.727877、H3 0.694360、H3-B 0.719658（H3-B 比 H3 +0.025298，比 H0 -0.008219）。结论：两个子群相对 H3 均改善，但仍未超过 H0。
10. **现有结果支持继续研究 L2-SP，还是应停止？** **建议停止把 L2-SP 作为提升泛化性能的方向。** H3-B 相对 H3 显著恢复（ΔAUROC +0.027712，CI [0.012805868685245515, 0.038153466582298276]），但仍低于 H0（Δ -0.011971，CI [-0.025593961775302886, 0.0011581733822822497] 跨 0），AUPRC 亦低于 H0 （0.413933 vs 0.428405），且 AP-only 与单图子群均未超过 H0；弱 L2-SP 只是减轻了强正则的损害，没有带来额外收益。若未来仍要研究，只建议作为正则化鲁棒性/多 seed 复核的低优先级项，不应再继续调 λ 或与 H2 组合。

## Per-epoch record completeness and limitations

- `training_history.csv` 记录原始 Train BCE、优化目标 `objective_loss`、Validation 原始 BCE、Macro-AUROC/AUPRC、L2-SP 原始项与 λR、Stage3/4 相对漂移、LR、BCE 梯度范数、L2-SP 单位梯度范数、更新次数与早停状态。
- 十类疾病逐 epoch Validation AUROC/AUPRC/BCE 记录在 `per_class_metrics.csv`；逐 epoch 逐类 Train/Valid BCE 在 `per_epoch_class_loss.csv`。
- `l2sp_gradient_history.csv` 由训练历史派生，补充 λ·‖g_SP‖ 与 λ·‖g_SP‖/‖g_BCE‖；梯度裁剪阈值为 1.0，但训练循环未单独保存每次更新是否触发裁剪，故该字段只记录阈值，不能伪造触发比例。
- `per_epoch_class_loss.csv` 的 Validation 逐 epoch 逐类 BCE 完整；Train 逐类 BCE 仅在最后一个 epoch 保留（分析重建步骤未保留训练时记录的全部中间 epoch 的 Train 逐类值），不伪造；Best/Last 的完整确定性 Train/Valid 逐类 BCE 见对应文件。
- 单 seed、Validation 重复使用、无 Test 标签；所有 CI 为当前官方 Validation 下的探索性统计。

