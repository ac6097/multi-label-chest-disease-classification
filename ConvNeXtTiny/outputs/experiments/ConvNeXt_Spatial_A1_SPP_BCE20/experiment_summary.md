# ConvNeXt spatial 实验摘要

## 实验状态

- 完成轮次：14 / 20
- 停止原因：按实验安排停止于第 14 个完整 epoch；原计划为 20 epochs
- Best Epoch：4
- Best Validation Macro-AUROC：0.779350
- Best Epoch Train Loss：0.304034
- Best Epoch Val Loss：0.344233
- Best Epoch Train-Val Gap：0.040200
- Best Epoch Macro-F1：0.242128
- Train-Val Loss Gap 定义为 Val Loss − Train Loss。Train 使用增强、Validation 不使用增强，因此 Loss 绝对值并非完全同分布，应结合趋势和 AUROC 判断泛化。

## 网络路径

ConvNeXt-Tiny ImageNet Backbone → Stage4 空间特征 → 1×1 + 2×2 固定 Adaptive Average Pooling 拼接 → LayerNorm → Linear(10)；Study logits 使用 Logit Mean。

## 最佳轮次逐类验证指标

| 疾病 | AUROC | AUPRC | Precision | Recall | F1@0.5 | TP | FP | TN | FN | 阳性数 | 阴性数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Enlarged Cardiomediastinum | 0.737500 | 0.071316 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 835 | 24 | 24 | 835 |
| Pneumothorax | 0.830654 | 0.254285 | 0.666667 | 0.054054 | 0.100000 | 2 | 1 | 821 | 35 | 37 | 822 |
| Consolidation | 0.728354 | 0.166395 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 813 | 46 | 46 | 813 |
| Pneumonia | 0.675746 | 0.150254 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 791 | 68 | 68 | 791 |
| Edema | 0.854617 | 0.391954 | 0.443038 | 0.304348 | 0.360825 | 35 | 44 | 700 | 80 | 115 | 744 |
| Cardiomegaly | 0.769086 | 0.428325 | 0.615385 | 0.135593 | 0.222222 | 24 | 15 | 667 | 153 | 177 | 682 |
| Atelectasis | 0.741174 | 0.408128 | 0.645161 | 0.103627 | 0.178571 | 20 | 11 | 655 | 173 | 193 | 666 |
| Lung Opacity | 0.714707 | 0.453453 | 0.629630 | 0.079439 | 0.141079 | 17 | 10 | 635 | 197 | 214 | 645 |
| Pleural Effusion | 0.881976 | 0.737039 | 0.679856 | 0.726923 | 0.702602 | 189 | 89 | 510 | 71 | 260 | 599 |
| No Finding | 0.859690 | 0.810363 | 0.654054 | 0.790850 | 0.715976 | 242 | 128 | 425 | 64 | 306 | 553 |

## 可复现性与产物

- Seed：42
- Optimizer：AdamW，Weight Decay=0.01
- Scheduler：optimizer_update 级 Linear Warmup + Cosine，warmup=2 epochs
- Cosine 结束 Epoch：20；最低 LR：Backbone=5e-07，Head=1e-05
- Early Stopping：disabled；最大训练轮数=20
- Backbone LR：1e-05；Head LR：0.0002
- 初始化：torchvision ImageNet ConvNeXt-Tiny feature weights with a fresh spatial classification head; resume is restricted to this run's own last checkpoint
- ImageNet 权重来源：ConvNeXt_Tiny_Weights.IMAGENET1K_V1
- ImageNet 权重 SHA-256：983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d
- Stage4 smoke 特征形状：[2, 768, 10, 10]
- 参数计数：GAP 参照=27,827,818，当前模型=27,864,682，相对 GAP 参照新增=36,864
- 训练峰值显存：1.437 GiB
- Smoke Test 峰值显存：0.587 GiB
- Best checkpoint：\\192.168.60.1\团队文件-Public\multi-label-chest-disease-classification\ConvNeXtTiny\outputs\experiments\ConvNeXt_Spatial_A1_SPP_BCE20\best.pth
- Last checkpoint：\\192.168.60.1\团队文件-Public\multi-label-chest-disease-classification\ConvNeXtTiny\outputs\experiments\ConvNeXt_Spatial_A1_SPP_BCE20\last.pth
- Epoch 20 checkpoint：\\192.168.60.1\团队文件-Public\multi-label-chest-disease-classification\ConvNeXtTiny\outputs\experiments\ConvNeXt_Spatial_A1_SPP_BCE20\epoch20.pth
- 其他产物：config.json、train_log.csv、per_class_metrics.csv、loss_curve.png、auroc_curve.png、per_class_auroc_curve.png、lr_curve.png。

## 历史 GAP 参照

历史 ConvNeXt GAP best Macro-AUROC=0.791286，Best Epoch=5。
该参照配置与当前实验不同；差值不能单独归因于某一个结构或训练因素。

## 分阶段趋势

| 阶段 | Epoch 数 | Train Loss 首→末 | Val Loss 首→末 | AUROC 首→末 | 阶段最佳 AUROC |
|---|---:|---:|---:|---:|---:|
| Warmup | 2 | 0.39528 → 0.33452 | 0.35522 → 0.34103 | 0.70939 → 0.76180 | 0.761797 (Epoch 2) |
| Cosine | 12 | 0.31734 → 0.17776 | 0.32999 → 0.40299 | 0.77449 → 0.74952 | 0.779350 (Epoch 4) |
