# R5 / R6 训练结果对照

R5 是已完成的 512×512 Stage3/4 Global-only 基线；R6 使用同一原始 DINOv3 通用视觉权重、同一 seed、Study 采样、BCE、优化器、学习率、调度和早停规则，独立解冻 Stem 与 Stage1/2。两者均以 Best checkpoint 进行验证比较。

- R5 Macro-AUROC / AUPRC：0.803514 / 0.428356
- R6 Macro-AUROC / AUPRC：0.798802 / 0.432019
- R6−R5 Macro-AUROC：-0.004712，95% CI [-0.014618, +0.002907]
- R6−R5 Macro-AUPRC：+0.003663，95% CI [-0.018225, +0.019995]
- Subject-cluster paired bootstrap：2000 次，Seed=42；859 个 Study / 228 个 Subject。
- 置信区间反映同一 Validation 上的配对抽样不确定性；最佳 checkpoint 已通过该 Validation 选择，且多类别比较为探索性结果。

## 训练过程对照

| Run | Best Epoch | Completed | Stop | Updates through Best / Stop | Mean images/s through Best | Peak reserved GiB | Trainable / Frozen Params |
|---|---:|---:|---|---:|---:|---:|---:|
| R5 | 6 | 9 | `three_condition_early_stop` | 6,037 / 9,056 | 22.14 | 1.57 | 26,592,778 / 1,235,040 |
| R6 | 6 | 9 | `three_condition_early_stop` | 6,037 / 9,056 | 22.08 | 4.04 | 27,827,818 / 0 |

Train/Validation Loss at Best and Final Epoch are recorded in `optimizer_updates_comparison.csv`; complete Epoch curves are in `train_log.csv` and `train_valid_loss.png`.

## 逐类别结果

| Disease | R5 AUROC | R6 AUROC | Δ AUROC (95% CI) | R5 AUPRC | R6 AUPRC | Δ AUPRC (95% CI) | Pos/Neg |
|---|---:|---:|---:|---:|---:|---:|---:|
| Enlarged Cardiomediastinum | 0.7077 | 0.7109 | +0.0032 [-0.0454, +0.0515] | 0.0969 | 0.0656 | -0.0314 [-0.1301, +0.0012] | 24/835 |
| Pneumothorax | 0.8554 | 0.8535 | -0.0019 [-0.0475, +0.0295] | 0.3509 | 0.4095 | +0.0586 [-0.0813, +0.1595] | 37/822 |
| Consolidation | 0.7885 | 0.7787 | -0.0098 [-0.0441, +0.0235] | 0.1997 | 0.2072 | +0.0075 [-0.0883, +0.0550] | 46/813 |
| Pneumonia | 0.7246 | 0.7175 | -0.0071 [-0.0566, +0.0276] | 0.2032 | 0.1967 | -0.0064 [-0.0556, +0.0777] | 68/791 |
| Edema | 0.8855 | 0.8742 | -0.0113 [-0.0244, -0.0005] | 0.4790 | 0.4725 | -0.0065 [-0.0603, +0.0568] | 115/744 |
| Cardiomegaly | 0.7769 | 0.7711 | -0.0058 [-0.0219, +0.0100] | 0.4305 | 0.4278 | -0.0027 [-0.0403, +0.0516] | 177/682 |
| Atelectasis | 0.7926 | 0.7910 | -0.0016 [-0.0147, +0.0121] | 0.4626 | 0.4637 | +0.0011 [-0.0278, +0.0349] | 193/666 |
| Lung Opacity | 0.7233 | 0.7242 | +0.0008 [-0.0169, +0.0195] | 0.4575 | 0.4638 | +0.0063 [-0.0354, +0.0580] | 214/645 |
| Pleural Effusion | 0.9087 | 0.9070 | -0.0017 [-0.0123, +0.0076] | 0.7764 | 0.8037 | +0.0273 [-0.0008, +0.0517] | 260/599 |
| No Finding | 0.8719 | 0.8599 | -0.0120 [-0.0252, +0.0014] | 0.8269 | 0.8098 | -0.0171 [-0.0346, +0.0003] | 306/553 |

所有逐类指标均由 R5/R6 各自 Best checkpoint 的 Study Logit Mean 概率计算；Study 按 `(Subject_id, Study_id)` 键严格对齐。Bootstrap 对同一 Subject 下所有 Study 成组重采样；单一类别在某次抽样中缺失阳性或阴性时跳过该类指标，宏平均使用可计算类别。

该比较用于估计全量微调相对于 R5 的表现差异，不能消除单次随机种子及 Validation 选模带来的不确定性。
