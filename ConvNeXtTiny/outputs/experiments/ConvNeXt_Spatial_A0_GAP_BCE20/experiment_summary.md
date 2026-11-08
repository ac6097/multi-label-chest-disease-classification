# ConvNeXt spatial 实验摘要

## 实验状态

- 完成轮次：12 / 20
- 停止原因：训练进行中；本文件为当前进度快照
- Best Epoch：8
- Best Validation Macro-AUROC：0.795693
- Best Epoch Train Loss：0.274934
- Best Epoch Val Loss：0.327185
- Best Epoch Train-Val Gap：0.052251
- Best Epoch Macro-F1：0.300177
- Train-Val Loss Gap 定义为 Val Loss − Train Loss。Train 使用增强、Validation 不使用增强，因此 Loss 绝对值并非完全同分布，应结合趋势和 AUROC 判断泛化。

## 网络路径

ConvNeXt-Tiny ImageNet Backbone → Stage4 空间特征 → GAP → LayerNorm → Linear(10)；Study logits 使用 Logit Mean。

## 最佳轮次逐类验证指标

| 疾病 | AUROC | AUPRC | Precision | Recall | F1@0.5 | TP | FP | TN | FN | 阳性数 | 阴性数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Enlarged Cardiomediastinum | 0.762375 | 0.121706 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 835 | 24 | 24 | 835 |
| Pneumothorax | 0.839104 | 0.237749 | 0.500000 | 0.108108 | 0.177778 | 4 | 4 | 818 | 33 | 37 | 822 |
| Consolidation | 0.751658 | 0.214639 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 813 | 46 | 46 | 813 |
| Pneumonia | 0.724967 | 0.226745 | 0.000000 | 0.000000 | 0.000000 | 0 | 1 | 790 | 68 | 68 | 791 |
| Edema | 0.873656 | 0.449863 | 0.522222 | 0.408696 | 0.458537 | 47 | 43 | 701 | 68 | 115 | 744 |
| Cardiomegaly | 0.755426 | 0.402052 | 0.397727 | 0.197740 | 0.264151 | 35 | 53 | 629 | 142 | 177 | 682 |
| Atelectasis | 0.750331 | 0.437017 | 0.481818 | 0.274611 | 0.349835 | 53 | 57 | 609 | 140 | 193 | 666 |
| Lung Opacity | 0.740191 | 0.483340 | 0.594937 | 0.219626 | 0.320819 | 47 | 32 | 613 | 167 | 214 | 645 |
| Pleural Effusion | 0.893130 | 0.760832 | 0.706349 | 0.684615 | 0.695313 | 178 | 74 | 525 | 82 | 260 | 599 |
| No Finding | 0.866099 | 0.817763 | 0.713846 | 0.758170 | 0.735341 | 232 | 93 | 460 | 74 | 306 | 553 |

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
- 参数计数：GAP 参照=27,827,818，当前模型=27,827,818，相对 GAP 参照新增=0
- 训练峰值显存：1.436 GiB
- Smoke Test 峰值显存：0.587 GiB
- Best checkpoint：\\192.168.60.1\团队文件-Public\multi-label-chest-disease-classification\ConvNeXtTiny\outputs\experiments\ConvNeXt_Spatial_A0_GAP_BCE20\best.pth
- Last checkpoint：\\192.168.60.1\团队文件-Public\multi-label-chest-disease-classification\ConvNeXtTiny\outputs\experiments\ConvNeXt_Spatial_A0_GAP_BCE20\last.pth
- Epoch 20 checkpoint：\\192.168.60.1\团队文件-Public\multi-label-chest-disease-classification\ConvNeXtTiny\outputs\experiments\ConvNeXt_Spatial_A0_GAP_BCE20\epoch20.pth
- 其他产物：config.json、train_log.csv、per_class_metrics.csv、loss_curve.png、auroc_curve.png、per_class_auroc_curve.png、lr_curve.png。

## 历史 GAP 参照

历史 ConvNeXt GAP best Macro-AUROC=0.791286，Best Epoch=5。
该参照配置与当前实验不同；差值不能单独归因于某一个结构或训练因素。

## 分阶段趋势

| 阶段 | Epoch 数 | Train Loss 首→末 | Val Loss 首→末 | AUROC 首→末 | 阶段最佳 AUROC |
|---|---:|---:|---:|---:|---:|
| Warmup | 2 | 0.45282 → 0.34228 | 0.37395 → 0.33854 | 0.64675 → 0.74502 | 0.745015 (Epoch 2) |
| Cosine | 10 | 0.32181 → 0.23703 | 0.32847 → 0.34310 | 0.77067 → 0.78448 | 0.795693 (Epoch 8) |
