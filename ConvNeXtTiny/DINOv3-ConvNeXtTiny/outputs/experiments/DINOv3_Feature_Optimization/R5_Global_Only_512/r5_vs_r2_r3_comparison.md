# R5 Global-only 与 R2/R3 对照分析

## 实验身份与对照条件

- R5 从原始通用视觉 DINOv3 ConvNeXt-Tiny 权重独立初始化；没有加载 R2/R3 监督微调参数。
- R5 Global Head 初始 SHA 与 R3 保存记录一致。网络仅含 Backbone 与 Linear(768,10)，不含 Local Head、alpha 或 Auxiliary Loss。
- R5 复用 R3 的 512 输入、whole-Study sampler、seed 42、image budget、accumulation、optimizer/LR、BF16、channels-last、增强、冻结策略和三条件早停。R3 的 BCE 使用融合 logits，R5 使用 Global logits；此训练目标差异是 Local 训练方案整体效应的一部分。
- R3 checkpoint 重推理关闭/启用 Local 的输出，仅用于 R3-Global 与 R3-Fused 比较；R5 Best 来自独立 Global-only 训练。

## 验证指标

| Model | Macro-AUROC | Macro-AUPRC | Best Epoch |
|---|---:|---:|---:|
| R2 · 512 GAP | 0.802116 | 0.421206 | 6 |
| R3 · Global-only inference | 0.804485 | 0.418469 | 6 |
| R3 · Global+Local | 0.804653 | 0.418717 | 6 |
| R5 · Independent Global-only | 0.803514 | 0.428356 | 6 |

- R3 Best Epoch / Macro-AUROC: 6 / 0.804653。
- R3-Global-only 重推理 Macro-AUROC: 0.804485。
- R5 Best Epoch / Macro-AUROC: 6 / 0.803514。
- R5 第 6 轮 Macro-AUROC（与 R3 Best Epoch 对照）：0.803514

## Subject-cluster paired Bootstrap

- 2000 次有效重复，Seed=42；859 Validation Study / 228 Subject；所有模型复用同一 Subject 抽样索引。CI 未做多重比较校正，且 Best Epoch 由同一 Validation 选择。

| Comparison | Macro-AUROC Δ [95% CI] | Macro-AUPRC Δ [95% CI] |
|---|---:|---:|
| R5 − R2 | +0.001398 [-0.006374, +0.008761] | +0.007149 [-0.009265, +0.026053] |
| R3-Global − R5 | +0.000971 [-0.006841, +0.008143] | -0.009887 [-0.031324, +0.007042] |
| R3-Fused − R5 | +0.001139 [-0.006724, +0.008329] | -0.009638 [-0.031582, +0.006963] |

### 逐类 AUROC 差值

| Disease | R5−R2 Δ [95% CI] | R3-Global−R5 Δ [95% CI] | R3-Fused−R5 Δ [95% CI] |
|---|---:|---:|---:|
| Enlarged Cardiomediastinum | -0.01352 [-0.05514, +0.02388] | +0.03520 [-0.00962, +0.09168] | +0.03558 [-0.00917, +0.09211] |
| Pneumothorax | +0.00266 [-0.02270, +0.02729] | -0.00306 [-0.02260, +0.01384] | -0.00273 [-0.02230, +0.01404] |
| Consolidation | +0.01703 [-0.00241, +0.04305] | -0.02163 [-0.05994, +0.00301] | -0.02195 [-0.06022, +0.00256] |
| Pneumonia | -0.00564 [-0.02544, +0.02180] | -0.00199 [-0.03446, +0.02628] | -0.00119 [-0.03360, +0.02720] |
| Edema | +0.00356 [-0.00439, +0.01029] | -0.00409 [-0.01106, +0.00378] | -0.00403 [-0.01112, +0.00388] |
| Cardiomegaly | -0.00299 [-0.01656, +0.01101] | +0.00096 [-0.01140, +0.01566] | +0.00090 [-0.01142, +0.01560] |
| Atelectasis | +0.00877 [-0.00564, +0.02493] | +0.00256 [-0.00917, +0.01420] | +0.00270 [-0.00900, +0.01442] |
| Lung Opacity | +0.00155 [-0.01427, +0.01458] | +0.00420 [-0.01145, +0.02087] | +0.00433 [-0.01149, +0.02116] |
| Pleural Effusion | +0.00126 [-0.00309, +0.00626] | -0.00054 [-0.00501, +0.00434] | -0.00035 [-0.00481, +0.00454] |
| No Finding | +0.00129 [-0.00652, +0.00814] | -0.00190 [-0.01036, +0.00612] | -0.00186 [-0.01038, +0.00627] |

## Optimizer Updates 与 sampler 对照

- R3 截至 Best Epoch 的正式 optimizer updates：6037；R5：6037。R5/R3 每轮更新数与学习率记录逐轮见 `optimizer_updates_comparison.csv`。
- R3/R5 image budget 匹配：True；gradient accumulation 匹配：True；共 18513 个 Train Study 的 epoch sampler 顺序哈希匹配：True。
- 两组共同完成的 Epoch 中 optimizer-update 数逐轮匹配：True；epoch 末学习率逐轮匹配：True。
- Runtime：budget=8，accumulation=4，effective images/update=30.5，effective Studies/update=18.5，channels-last=True。

## 解释与后续建议

- R3-Global 与 R5 的差异是 R3 Global 路径经历融合 logits 监督的整体训练效应与 R5 纯 Global 监督之间的对照，仍可能受随机性、Best Epoch 选择和其他训练轨迹因素影响；即使区间排除 0，也不能单独归因为 Local 特征对 Backbone 的因果改造。
- 若 R3-Global−R5 区间跨 0，现有证据不足以认定 Local 训练路径带来稳定收益；不应仅因小幅点差停止或继续 Local 研究。若区间持续正向且更新预算、初始化及 Study 顺序均匹配，结果才值得后续用独立 seed/控制 loss 设计验证。
- R5 与 R2 的比较用于判断严格匹配 R3 runtime 后的独立 Global-only 水平；模型选择始终以 Validation 为准，Test 未用于本实验。
