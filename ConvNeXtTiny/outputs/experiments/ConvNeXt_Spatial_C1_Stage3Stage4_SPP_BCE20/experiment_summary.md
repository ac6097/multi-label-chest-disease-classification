# ConvNeXt spatial 实验摘要

## 实验状态

- 完成轮次：11 / 20
- 停止原因：按用户要求在第 12 轮训练期间停止；共完成 11 个完整 epoch，原计划为 20 epochs
- Best Epoch：6
- Best Validation Macro-AUROC：0.777955
- Best Epoch Train Loss：0.283722
- Best Epoch Val Loss：0.330113
- Best Epoch Train-Val Gap：0.046391
- Best Epoch Macro-F1：0.287536
- Train-Val Loss Gap 定义为 Val Loss − Train Loss。Train 使用增强、Validation 不使用增强，因此 Loss 绝对值并非完全同分布，应结合趋势和 AUROC 判断泛化。

## 网络路径

ConvNeXt-Tiny ImageNet Backbone → Stage3/Stage4 1×1 投影 → Stage4 双线性尺寸对齐 → 拼接 → 1×1 卷积融合 → GroupNorm/GELU → 1×1 + 2×2 固定 Adaptive Average Pooling 拼接 → LayerNorm → Linear(10)；Study logits 使用 Logit Mean。

## 最佳轮次逐类验证指标

| 疾病 | AUROC | AUPRC | Precision | Recall | F1@0.5 | TP | FP | TN | FN | 阳性数 | 阴性数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Enlarged Cardiomediastinum | 0.696956 | 0.058600 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 835 | 24 | 24 | 835 |
| Pneumothorax | 0.825081 | 0.267362 | 0.625000 | 0.135135 | 0.222222 | 5 | 3 | 819 | 32 | 37 | 822 |
| Consolidation | 0.738235 | 0.197609 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 813 | 46 | 46 | 813 |
| Pneumonia | 0.674240 | 0.137096 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 791 | 68 | 68 | 791 |
| Edema | 0.862331 | 0.414719 | 0.460177 | 0.452174 | 0.456140 | 52 | 61 | 683 | 63 | 115 | 744 |
| Cardiomegaly | 0.764489 | 0.427377 | 0.377246 | 0.355932 | 0.366279 | 63 | 104 | 578 | 114 | 177 | 682 |
| Atelectasis | 0.758499 | 0.454548 | 0.493333 | 0.191710 | 0.276119 | 37 | 38 | 628 | 156 | 193 | 666 |
| Lung Opacity | 0.705060 | 0.408565 | 0.470588 | 0.074766 | 0.129032 | 16 | 18 | 627 | 198 | 214 | 645 |
| Pleural Effusion | 0.894629 | 0.751380 | 0.720833 | 0.665385 | 0.692000 | 173 | 67 | 532 | 87 | 260 | 599 |
| No Finding | 0.860030 | 0.806823 | 0.779412 | 0.692810 | 0.733564 | 212 | 60 | 493 | 94 | 306 | 553 |

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
- 参数计数：GAP 参照=27,827,818，当前模型=28,260,458，相对 GAP 参照新增=432,640
- 训练峰值显存：1.469 GiB
- Smoke Test 峰值显存：0.592 GiB
- Best checkpoint：\\192.168.60.1\团队文件-Public\multi-label-chest-disease-classification\ConvNeXtTiny\outputs\experiments\ConvNeXt_Spatial_C1_Stage3Stage4_SPP_BCE20\best.pth
- Last checkpoint：\\192.168.60.1\团队文件-Public\multi-label-chest-disease-classification\ConvNeXtTiny\outputs\experiments\ConvNeXt_Spatial_C1_Stage3Stage4_SPP_BCE20\last.pth
- Epoch 20 checkpoint：\\192.168.60.1\团队文件-Public\multi-label-chest-disease-classification\ConvNeXtTiny\outputs\experiments\ConvNeXt_Spatial_C1_Stage3Stage4_SPP_BCE20\epoch20.pth
- 其他产物：config.json、train_log.csv、per_class_metrics.csv、loss_curve.png、auroc_curve.png、per_class_auroc_curve.png、lr_curve.png。

## 历史 GAP 参照

历史 ConvNeXt GAP best Macro-AUROC=0.791286，Best Epoch=5。
该参照配置与当前实验不同；差值不能单独归因于某一个结构或训练因素。

## 分阶段趋势

| 阶段 | Epoch 数 | Train Loss 首→末 | Val Loss 首→末 | AUROC 首→末 | 阶段最佳 AUROC |
|---|---:|---:|---:|---:|---:|
| Warmup | 2 | 0.38528 → 0.33578 | 0.36253 → 0.34077 | 0.70707 → 0.75643 | 0.756427 (Epoch 2) |
| Cosine | 9 | 0.31826 → 0.21254 | 0.32758 → 0.38702 | 0.77315 → 0.75967 | 0.777955 (Epoch 6) |
