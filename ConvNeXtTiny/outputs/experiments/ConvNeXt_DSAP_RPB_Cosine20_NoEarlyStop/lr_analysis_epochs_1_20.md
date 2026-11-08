# ConvNeXt-DSAP-RPB 学习率实验分析（Epoch 1–20）

## 实验结论

本实验从 torchvision ImageNet 权重和全新 DSAP-RPB head 初始化，按计划使用 2 轮 warmup，随后在 Epoch 3–20 进行余弦退火。Epoch 20 将 backbone/head 学习率降至设定最低值 `5e-7 / 1e-5`。根据用户要求，完成 Epoch 20 后停止；配置原计划为 40 轮，因此本文分析的是前 20 轮，不代表完整 40 轮实验。

20 轮中最佳验证 Macro-AUROC 为 **0.787977（Epoch 7）**。Epoch 20 达到最低学习率后，Macro-AUROC 为 **0.765762**，比最佳轮低 `0.022215`；验证 loss 从 Epoch 5 的最低值 `0.323914` 上升到 Epoch 20 的 `0.401027`，train–val loss gap 从 Epoch 7 的 `0.038810` 扩大到 `0.189083`。因此，降低学习率按曲线正常执行，但前 20 轮结果没有显示它扭转验证 AUROC 下滑或减轻过拟合。

相对上一轮 DSAP-RPB 实验的最佳 Macro-AUROC `0.785822`，本实验提高 `0.002155`。相对历史 GAP ConvNeXt 参照 `0.791286`，本实验低 `0.003310`。前者的提升是本次观察到的结果；不同 run 的训练轮数和初始化/配置差异限制了因果归因。

## 执行信息

| 项目 | 设置/结果 |
|---|---|
| 模型 | ConvNeXt-Tiny + DSAP-RPB；Study Logit Mean |
| 初始化 | torchvision ImageNet Backbone 权重；DSAP-RPB head 全新初始化 |
| Seed | 42 |
| 训练 / 验证数据 | `cxr_first/train.csv` / `cxr_first/valid.csv`；未访问 test split |
| Loss / weighting | BCE / sqrt weighting；`pos_weight_cap=5` |
| Peak LR（backbone / head） | `1e-5 / 2e-4` |
| Minimum LR（backbone / head） | `5e-7 / 1e-5` |
| 调度 | Warmup 2 epochs；optimizer-update 级 cosine，Epoch 20 到最低 LR |
| Early stopping | 关闭 |
| 实际完成 | 20 epochs；20,123 optimizer updates；skipped updates = 0 |
| 累积训练 / 验证时间 | 7.341 h / 0.547 h（日志计时） |
| 峰值 GPU 显存 | 1.564 GiB |
| 停止点 | Epoch 20 的 train/validation 指标、last checkpoint 和 milestone checkpoint 保存后停止 |

训练中断过一次：DataLoader 报 `FileNotFoundError`。恢复后检查 train/valid 清单共 31,766 张图片，缺失 0 张；随后从本实验的 Epoch 19 `last.pth` 恢复并完成 Epoch 20。未更换数据、模型或训练方案。

## 逐轮训练与验证指标

`train–val gap = val_loss − train_loss`。训练集使用数据增强、验证集不使用，因此该差值主要用于观察趋势，不应直接当作严格同分布的泛化误差。

| Epoch | Train loss | Val loss | Gap | Macro-AUROC | Macro-F1 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.441383 | 0.358181 | -0.083202 | 0.683322 | 0.104982 |
| 2 | 0.342458 | 0.341112 | -0.001346 | 0.727551 | 0.192919 |
| 3 | 0.324558 | 0.332437 | 0.007878 | 0.752780 | 0.243080 |
| 4 | 0.313253 | 0.342423 | 0.029170 | 0.762828 | 0.236857 |
| 5 | 0.304793 | 0.323914 | 0.019121 | 0.779958 | 0.241467 |
| 6 | 0.296673 | 0.323936 | 0.027263 | 0.779109 | 0.259511 |
| 7 | 0.289214 | 0.328024 | 0.038810 | **0.787977** | 0.246033 |
| 8 | 0.280802 | 0.344971 | 0.064169 | 0.776664 | 0.280770 |
| 9 | 0.271606 | 0.337900 | 0.066293 | 0.774194 | 0.297165 |
| 10 | 0.262862 | 0.338188 | 0.075326 | 0.783398 | 0.322777 |
| 11 | 0.255723 | 0.348728 | 0.093005 | 0.778080 | 0.302507 |
| 12 | 0.247139 | 0.352182 | 0.105043 | 0.774514 | 0.317257 |
| 13 | 0.239518 | 0.363137 | 0.123620 | 0.771656 | 0.314706 |
| 14 | 0.233105 | 0.374368 | 0.141264 | 0.770838 | 0.309438 |
| 15 | 0.228125 | 0.375422 | 0.147297 | 0.769787 | 0.335316 |
| 16 | 0.223145 | 0.384865 | 0.161720 | 0.767389 | 0.325851 |
| 17 | 0.218368 | 0.393314 | 0.174946 | 0.767260 | 0.336937 |
| 18 | 0.215961 | 0.393766 | 0.177805 | 0.767318 | 0.336027 |
| 19 | 0.213904 | 0.396882 | 0.182978 | 0.766771 | **0.342904** |
| 20 | 0.211944 | 0.401027 | 0.189083 | 0.765762 | 0.338131 |

## 学习率轨迹抽样

表内为每轮首个 optimizer update 使用的 backbone LR，以及该轮结束后准备给下一 update 的 LR。Head LR 始终是 backbone LR 的 20 倍。

| Epoch | Backbone LR：start → next | Head LR：start → next | 阶段 |
|---:|---:|---:|---|
| 1 | `4.968e-9 → 4.998e-6` | `9.935e-8 → 9.995e-5` | Warmup |
| 2 | `4.998e-6 → 1.000e-5` | `9.995e-5 → 2.000e-4` | Warmup 完成 |
| 3 | `1.000e-5 → 9.928e-6` | `2.000e-4 → 1.986e-4` | Cosine |
| 7 | `8.889e-6 → 8.303e-6` | `1.778e-4 → 1.661e-4` | Cosine；最佳 AUROC |
| 10 | `6.874e-6 → 6.075e-6` | `1.375e-4 → 1.215e-4` | Cosine |
| 15 | `2.874e-6 → 2.196e-6` | `5.748e-5 → 4.393e-5` | Cosine |
| 19 | `7.861e-7 → 5.722e-7` | `1.572e-5 → 1.144e-5` | Cosine 接近终点 |
| 20 | `5.722e-7 → 5.000e-7` | `1.144e-5 → 1.000e-5` | 到达最低 LR |

实际日志与预览的每轮 optimizer update 数不固定，单轮为 1004–1009；调度按每轮实际确定的 sampler update 数构建。Epoch 20 的 `lr_last_used` 为 backbone `5e-7`、head `1e-5`，`lr_next` 保持这两个最低值。

## 结果解读

1. **最佳点出现较早。** AUROC 从 Epoch 1 上升至 Epoch 7 的峰值；余弦继续降低学习率期间没有超过该峰值。Epoch 10 的 `0.783398` 是后续最高点，但仍低于 Epoch 7。
2. **训练拟合持续改善，验证 loss 持续恶化。** Train loss 从 `0.441383` 降至 `0.211944`；验证 loss 在 Epoch 5 最低，此后总体上升至 Epoch 20 的最高值。Gap 从 Epoch 7 的 `0.038810` 扩大至 `0.189083`，和验证 AUROC 长期回落一致，显示明显的训练/验证分化。
3. **F1 与 AUROC 走势不同。** Macro-F1 在 Epoch 19 达到 `0.342904`，而该轮 AUROC 为 `0.766771`。阈值下的 F1 提升不能替代排序能力指标 AUROC；模型选择仍按实验计划中的 Macro-AUROC best checkpoint。
4. **本阶段尚不能验证固定低 LR 的长期效果。** 用户要求在 Epoch 20 停止；配置原定 Epoch 21–40 保持最低 LR，这 20 轮未执行，因此不能对固定低 LR 阶段的收益作结论。

## 参照结果

| 参照 | Best Epoch | Macro-AUROC | 本实验差值 |
|---|---:|---:|---:|
| 本次 DSAP-RPB cosine20 | 7 | 0.787977 | — |
| 上一轮 DSAP-RPB | 7 | 0.785822 | +0.002155 |
| 历史 ConvNeXt GAP | 5 | 0.791286 | -0.003310 |

这些是验证集比较。GAP 参照使用的 backbone LR 与本实验不同，不能把差值单独解释为结构或学习率造成的因果效果；test split 未参与本次选择或评估。

## 产物与可复现性

- 逐轮原始统计：`train_log.csv`
- 本次 best（Epoch 7）：`best.pth`、`best.json`
- Epoch 20 last：`last.pth`、`last.json`
- Epoch 20 milestone：`epoch20.pth`
- 曲线：`loss_curve.png`、`auroc_curve.png`、`lr_curve.png`、`per_class_auroc_curve.png`
- 配置快照：`config.json`；resume 来源为本实验 Epoch 19 `last.pth`

训练最终停止在 Epoch 20，且进程树已退出。由于用户要求在 Epoch 20 停止，后续 Epoch 21–40 和 test 集评估均未运行。
