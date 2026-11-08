# DenseNet-121 448×448 微调结果

## 训练定义

- 初始化：DenseNet121/checkpoint/best/best_by_macro_auc.pth，Epoch 6，SHA-256 为 8162986883188d168b013ad926f70eccddd559f61d9c922b05ab5db9eff7d477。
- 加载：完整模型 strict=True；Smoke Test 确认真实 448 Study 输出 [1,10] logits，并完成 BCE backward 与 AdamW step。
- 唯一实验变量：输入分辨率 448。DenseNet-121、标准 GAP、Linear(1024,10)、Standard BCE、Logit Mean、轻量增强和 Study 级损失保持不变；未启用 multiscale、CBAM、Label Cross-Attention、Weighted BCE 或 ASL。
- 配置：seed 42，16/2，AdamW，backbone/head LR 5e-6 / 5e-5，weight decay 1e-4，gradient clip 1.0，warmup 2，最多 8 epoch，patience 4，首轮冻结 backbone，BF16，4 workers。
- 320 匹配对照已存在，使用同一 Epoch 6 权重、BCE、optimizer、学习率、16/2、seed、增强、workers 与早停参数，因此不重复训练 320。
- validation：859 Study、1,422 张图；未读取测试集标签或提交预测。

## Epoch 轨迹

| Epoch | Train loss | Val loss | Macro-AUROC | Train 秒 | Val 秒 | Peak allocated GiB |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.31767 | 0.33784 | 0.77087 | 1,352.8 | 89.4 | 0.379 |
| **2** | **0.30051** | **0.32276** | **0.78688** | 1,336.0 | 90.4 | 4.145 |
| 3 | 0.29101 | 0.32400 | 0.78616 | 1,350.6 | 108.7 | 4.145 |
| 4 | 0.28402 | 0.32528 | 0.78583 | 1,346.0 | 77.8 | 4.145 |
| 5 | 0.28076 | 0.33004 | 0.78604 | 1,335.9 | 89.7 | 4.145 |
| 6 | 0.27468 | 0.33153 | 0.78622 | 1,330.2 | 87.8 | 4.145 |

最佳 checkpoint 为 Epoch 2，Macro-AUROC 0.78688。Epoch 3–6 连续四轮未超过 best，训练按 patience 4 停止；共训练 6 轮。训练 loss 持续下降，验证 loss 在 Epoch 2 的 0.32276 最低后上升，Macro-AUROC 也在 Epoch 2 达峰，显示后续轮次泛化收益停止并开始退化。

Pneumonia 的 AUROC 在被选为 best 的 Epoch 2 为 0.68460；后续 Epoch 5 曾升到 0.70234，但该轮整体 Macro-AUROC 低于 best，不能据此挑选一个非 best checkpoint。Lung Opacity 在 best Epoch 2 为 0.71332。

## 产物

- Best checkpoint：[best_by_macro_auc.pth](../DenseNet121/outputs/experiments/densenet_448_e6/checkpoint/best/best_by_macro_auc.pth)
- Best checkpoint SHA-256：f99185930e54955ac93640e115bbb4cb3ebe128711096db2b9303f68369b1c1a
- 文件大小：28,463,825 bytes（约 28.46 MB）。
- epoch 指标和逐类指标：DenseNet121/outputs/experiments/densenet_448_e6/outputs/logs/train.log
- Smoke Test 结果：DenseNet121/outputs/experiments/densenet_448_smoke/smoke_result.json
- 曲线、逐类比较、bootstrap、Grad-CAM、延迟：见 [320/448 对照目录](../DenseNet121/outputs/experiments/densenet_448_analysis_final/comparison_summary.json)。
