# DenseNet-121 当前优化决策

## 已完成的多尺度 GAP 对照

按 320×320、Dense Block2/3/4 GAP、Logit Mean、Standard BCE 完成 8 epoch 配对结构实验。Epoch 6 best backbone 严格迁移，分类头重新初始化；best 出现在实验 epoch 7。

| 模型 | Validation Macro-AUROC | 差值 vs Epoch 6 基线 |
|---|---:|---:|
| DenseNet-121 GAP，Epoch 6 | 0.788776 | — |
| DenseNet-121 Multi-scale GAP，epoch 7 | 0.770401 | -0.018375 |

单图 Study 为 0.6952 vs 基线 0.7106；多图 Study 为 0.8156 vs 基线 0.8345。除 Consolidation 外，多尺度模型其余九类 AUROC 均低于基线。详细指标、配置、checkpoint SHA、训练耗时及本机延迟见 [`densenet_multiscale_gap_results.md`](densenet_multiscale_gap_results.md)。

## 当前决定

**不采用多尺度 GAP 替换当前基线。** 保留 [Epoch 6 best checkpoint](../DenseNet121/checkpoint/best/best_by_macro_auc.pth) 作为当前可靠模型；多尺度 checkpoint 仅作为独立实验结果保留在 `DenseNet121/outputs/experiments/multiscale_gap_320_e6/`。本地 RTX 4060 上的 Study 前向延迟已测，T4 延迟和官方 100 ms 计时单位尚未验证。

本次仅执行用户指定的多尺度实验。448×448 分辨率、CBAM、Label Cross-Attention 及其他训练均未启动；后续实验需单独确定设置并明确启动。
