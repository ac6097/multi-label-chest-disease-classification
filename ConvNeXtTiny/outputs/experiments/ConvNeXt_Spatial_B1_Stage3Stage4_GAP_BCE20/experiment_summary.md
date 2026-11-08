# ConvNeXt spatial 实验摘要

## 实验状态

- 完成轮次：16 / 20
- 停止原因：按实验安排停止于第 16 个完整 epoch；原计划为 20 epochs
- Best Epoch：6
- Best Validation Macro-AUROC：0.783970
- Best Epoch Train Loss：0.291384
- Best Epoch Val Loss：0.321078
- Best Epoch Train-Val Gap：0.029694
- Best Epoch Macro-F1：0.268479
- Train-Val Loss Gap 定义为 Val Loss − Train Loss。Train 使用增强、Validation 不使用增强，因此 Loss 绝对值并非完全同分布，应结合趋势和 AUROC 判断泛化。

## 网络路径

ConvNeXt-Tiny ImageNet Backbone → Stage3/Stage4 1×1 投影 → Stage4 双线性尺寸对齐 → 拼接 → 1×1 卷积融合 → GroupNorm/GELU → GAP → LayerNorm → Linear(10)；Study logits 使用 Logit Mean。

## 最佳轮次逐类验证指标

| 疾病 | AUROC | AUPRC | Precision | Recall | F1@0.5 | TP | FP | TN | FN | 阳性数 | 阴性数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Enlarged Cardiomediastinum | 0.681337 | 0.087760 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 835 | 24 | 24 | 835 |
| Pneumothorax | 0.832084 | 0.245773 | 0.200000 | 0.027027 | 0.047619 | 1 | 4 | 818 | 36 | 37 | 822 |
| Consolidation | 0.737432 | 0.154245 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 813 | 46 | 46 | 813 |
| Pneumonia | 0.688955 | 0.163567 | 0.000000 | 0.000000 | 0.000000 | 0 | 0 | 791 | 68 | 68 | 791 |
| Edema | 0.871955 | 0.452565 | 0.524752 | 0.460870 | 0.490741 | 53 | 48 | 696 | 62 | 115 | 744 |
| Cardiomegaly | 0.774430 | 0.440686 | 0.445378 | 0.299435 | 0.358108 | 53 | 66 | 616 | 124 | 177 | 682 |
| Atelectasis | 0.767470 | 0.461950 | 0.540984 | 0.170984 | 0.259843 | 33 | 28 | 638 | 160 | 193 | 666 |
| Lung Opacity | 0.720093 | 0.455336 | 0.600000 | 0.056075 | 0.102564 | 12 | 8 | 637 | 202 | 214 | 645 |
| Pleural Effusion | 0.898950 | 0.770856 | 0.722892 | 0.692308 | 0.707269 | 180 | 69 | 530 | 80 | 260 | 599 |
| No Finding | 0.866991 | 0.816835 | 0.746479 | 0.692810 | 0.718644 | 212 | 72 | 481 | 94 | 306 | 553 |

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
- 参数计数：GAP 参照=27,827,818，当前模型=28,248,170，相对 GAP 参照新增=420,352
- 训练峰值显存：1.573 GiB
- Smoke Test 峰值显存：0.591 GiB
- Best checkpoint：\\192.168.60.1\团队文件-Public\multi-label-chest-disease-classification\ConvNeXtTiny\outputs\experiments\ConvNeXt_Spatial_B1_Stage3Stage4_GAP_BCE20\best.pth
- Last checkpoint：\\192.168.60.1\团队文件-Public\multi-label-chest-disease-classification\ConvNeXtTiny\outputs\experiments\ConvNeXt_Spatial_B1_Stage3Stage4_GAP_BCE20\last.pth
- Epoch 20 checkpoint：\\192.168.60.1\团队文件-Public\multi-label-chest-disease-classification\ConvNeXtTiny\outputs\experiments\ConvNeXt_Spatial_B1_Stage3Stage4_GAP_BCE20\epoch20.pth
- 其他产物：config.json、train_log.csv、per_class_metrics.csv、loss_curve.png、auroc_curve.png、per_class_auroc_curve.png、lr_curve.png。

## 历史 GAP 参照

历史 ConvNeXt GAP best Macro-AUROC=0.791286，Best Epoch=5。
该参照配置与当前实验不同；差值不能单独归因于某一个结构或训练因素。

## 分阶段趋势

| 阶段 | Epoch 数 | Train Loss 首→末 | Val Loss 首→末 | AUROC 首→末 | 阶段最佳 AUROC |
|---|---:|---:|---:|---:|---:|
| Warmup | 2 | 0.40175 → 0.33835 | 0.36131 → 0.33847 | 0.69416 → 0.74688 | 0.746878 (Epoch 2) |
| Cosine | 14 | 0.32014 → 0.20003 | 0.32491 → 0.40422 | 0.77111 → 0.76245 | 0.783970 (Epoch 6) |
