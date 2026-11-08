# DenseNet-121 320×320 与 448×448 对照

## 数据与比较口径

三套预测均覆盖相同的 validation 859 Study、1,422 张图，标签逐行校验一致。使用 Epoch 6 原始模型、已有 320 Standard BCE 续训 best、448 续训 best 的保存验证预测；测试集没有参与。

原始 Epoch 6 的 checkpoint SHA-256 为 8162986883188d168b013ad926f70eccddd559f61d9c922b05ab5db9eff7d477。历史文档中的 fresh inference Macro-AUROC 为 0.788776；本次使用训练保存的 val_best.csv 重算为 0.788827，差约 0.00005，与已记录的 BF16 Study 聚合差异一致。

## Macro-AUROC 与 Subject Bootstrap

| 模型 | Macro-AUROC | 相对 Epoch 6 | 最佳 epoch |
|---|---:|---:|---:|
| Epoch 6 原始 320 | 0.78883 | — | 6 |
| 匹配 320 Standard BCE continuation | 0.78927 | +0.00044 | 1 |
| 448 Standard BCE continuation | 0.78688 | -0.00194 | 2 |

2,000 次、seed 42、按 Subject 成簇的配对 bootstrap：

| 差值 | Bootstrap 均值 | 95% CI | CI 是否跨 0 |
|---|---:|---:|---|
| 448 − 匹配 320 | -0.00240 | [-0.01293, 0.00655] | 是 |
| 448 − Epoch 6 | -0.00197 | [-0.01259, 0.00701] | 是 |
| 匹配 320 − Epoch 6 | +0.00043 | [-0.00026, 0.00112] | 是 |

整体变化小且区间跨 0；目前没有证据支持 448 提升 Macro-AUROC。

## 十类 AUROC / AUPRC

| 类别 | Epoch 6 AUC / AP | 匹配 320 AUC / AP | 448 AUC / AP | 448 − 320 AUC |
|---|---:|---:|---:|---:|
| Enlarged Cardiomediastinum | 0.7515 / 0.0847 | 0.7500 / 0.0841 | 0.7301 / 0.0626 | -0.0200 |
| Pneumothorax | 0.8298 / 0.2769 | 0.8317 / 0.2811 | 0.8440 / 0.2437 | +0.0123 |
| Consolidation | 0.7530 / 0.1921 | 0.7534 / 0.1935 | 0.7524 / 0.1544 | -0.0009 |
| Pneumonia | 0.6926 / 0.1778 | 0.6927 / 0.1812 | 0.6846 / 0.1616 | -0.0081 |
| Edema | 0.8646 / 0.4332 | 0.8646 / 0.4324 | 0.8649 / 0.4062 | +0.0003 |
| Cardiomegaly | 0.7844 / 0.4350 | 0.7862 / 0.4395 | 0.7718 / 0.4141 | -0.0144 |
| Atelectasis | 0.7635 / 0.4406 | 0.7635 / 0.4428 | 0.7502 / 0.4223 | -0.0133 |
| Lung Opacity | 0.6986 / 0.4040 | 0.7002 / 0.4048 | 0.7133 / 0.4380 | +0.0132 |
| Pleural Effusion | 0.8908 / 0.7844 | 0.8911 / 0.7847 | 0.8929 / 0.7834 | +0.0017 |
| No Finding | 0.8592 / 0.7969 | 0.8593 / 0.7964 | 0.8646 / 0.8015 | +0.0053 |

Pneumothorax 和 Lung Opacity 的 AUROC 提高；Pneumonia 的 best-checkpoint AUROC/AUPRC 均低于 320。Consolidation 基本持平但 AUPRC 下降。Enlarged Cardiomediastinum、Cardiomegaly、Atelectasis 有较明显 AUROC 下降。448 的类间收益不一致。

## 单图与多图 Study

| 分组 | Study 数 | Epoch 6 | 匹配 320 | 448 |
|---|---:|---:|---:|---:|
| 单图 Study | 391 | 0.71061 | 0.71128 | 0.71265 |
| 多图 Study | 468 | 0.83457 | 0.83529 | 0.83015 |

448 在单图组只有轻微提高，在多图组低于两个 320 模型。

## 训练曲线与 Grad-CAM

[训练/验证 loss 与 Macro-AUROC 曲线](../DenseNet121/outputs/experiments/densenet_448_analysis_final/training_curves.png)显示 448 在 Epoch 1 较低，Epoch 2 接近 320 best，此后未刷新；train loss 继续下降而 val loss 回升。

[逐类 AUROC 图](../DenseNet121/outputs/experiments/densenet_448_analysis_final/class_auroc.png)和 [六个弱类 validation 案例的 Grad-CAM 对照](../DenseNet121/outputs/experiments/densenet_448_analysis_final/gradcam_320_vs_448.png)用于描述模型关注区域差异。Grad-CAM 案例按 validation 上的 448/320 概率差选择，不构成显著性或因果证据；案例列表见同目录 gradcam_case_selection.csv。

## 推理成本

本地 RTX 4060 Laptop；validation 中按相同顺序预热 10 个、测量 100 个 Study，平均每 Study 1.67 张图。延迟是本机同步计时，不代表 T4。

| 模型 | 图像尺寸 | 单图 forward mean / P95 | 单图 decode+transform+H2D+forward mean / P95 | Study forward+fusion mean / P95 | Study end-to-end mean / P50 / P95 | 文件大小 |
|---|---:|---:|---:|---:|---:|---:|
| Epoch 6 原始 | 320 | 33.26 / 36.83 ms | 126.82 / 155.40 ms | 44.32 / 57.21 ms | 142.02 / 125.69 / 192.08 ms | 28.46 MB |
| 匹配 continuation | 320 | 32.48 / 34.86 ms | 111.49 / 126.61 ms | 43.37 / 59.39 ms | 140.47 / 125.08 / 197.11 ms | 28.46 MB |
| continuation | 448 | 31.21 / 36.71 ms | 113.55 / 131.07 ms | 51.33 / 79.56 ms | 156.35 / 148.93 / 231.98 ms | 28.46 MB |

448 Study decode/preprocess 均值约 102.0ms（320 约 94.8ms），H2D 约 2.11ms，forward/fusion 均值约 51.3ms。Study end-to-end 均值比匹配 320 高约 11.3%，P95 高约 17.7%。三种模型文件均远低于 500MB。官方 T4 16GB/100ms 约束尚未在 T4 或官方计时环境验证；本机 4060 的单图端到端均值超过 100ms，不能据此判断 T4 结果。
