# ConvNeXt DSAP-RPB 实验摘要

## 实验状态

- 完成轮次：20 / 40
- 停止原因：按用户要求，在 Epoch 20 指标及 checkpoint 保存后停止；配置原计划 40 轮，本次只完成前 20 轮
- Best Epoch：7
- Best Validation Macro-AUROC：0.787977
- Best Epoch Train Loss：0.289214
- Best Epoch Val Loss：0.328024
- Best Epoch Train-Val Gap：0.038810
- Best Epoch Macro-F1：0.246033
- Train-Val Loss Gap 定义为 Val Loss − Train Loss。Train 使用增强、Validation 不使用增强，因此 Loss 绝对值并非完全同分布，应结合趋势和 AUROC 判断泛化。

## 网络路径

ConvNeXt-Tiny ImageNet Backbone Stage4 → channel projection → 二维正弦余弦位置编码 → 带二维 Relative Position Bias 的多头空间 Self-Attention → 十个 Disease Query 空间池化 → raw GAP + sigmoid(alpha_k) × disease attention feature → 十个独立疾病分类器。Study 输出继续对图像 Logits 做 Logit Mean。

- Self-Attention 的 Query/Key 来自加入位置编码的 Token，Value 来自投影后的原始 Token；Relative Position Bias 按 (Δh, Δw) 查表并加到 QK logits。
- Position Encoding 是固定二维正弦余弦值，只有缩放参数可训练；Disease Query、Relative Position Bias、投影层和每类 alpha/分类器均参与优化。

## 最佳轮次逐类 AUROC

| 疾病 | AUROC | AUPRC | F1 | 阳性数 | 阴性数 |
|---|---:|---:|---:|---:|---:|
| Enlarged Cardiomediastinum | 0.741916 | 0.135784 | 0.000000 | 24 | 835 |
| Pneumothorax | 0.761952 | 0.182036 | 0.051282 | 37 | 822 |
| Consolidation | 0.757661 | 0.126319 | 0.000000 | 46 | 813 |
| Pneumonia | 0.730312 | 0.169179 | 0.000000 | 68 | 791 |
| Edema | 0.862599 | 0.400554 | 0.174497 | 115 | 744 |
| Cardiomegaly | 0.760757 | 0.419726 | 0.255319 | 177 | 682 |
| Atelectasis | 0.780042 | 0.497082 | 0.350746 | 193 | 666 |
| Lung Opacity | 0.716967 | 0.434021 | 0.208955 | 214 | 645 |
| Pleural Effusion | 0.901175 | 0.785740 | 0.680761 | 260 | 599 |
| No Finding | 0.866385 | 0.808320 | 0.738769 | 306 | 553 |

## 可学习融合系数

- Best checkpoint alpha：Enlarged Cardiomediastinum=0.106926, Pneumothorax=0.109623, Consolidation=0.107995, Pneumonia=0.106364, Edema=0.109333, Cardiomegaly=0.107963, Atelectasis=0.107572, Lung Opacity=0.105718, Pleural Effusion=0.111496, No Finding=0.108926

## 可复现性与产物

- Seed：42
- Optimizer：AdamW，Weight Decay=0.01
- Scheduler：optimizer_update 级 Linear Warmup + Cosine，warmup=2 epochs
- Cosine 结束 Epoch：20；最低 LR：Backbone=5e-07，Head=1e-05
- Early Stopping：disabled；最大训练轮数=40
- Backbone LR：1e-05；Head LR：0.0002
- 初始化：Original run started from torchvision ImageNet ConvNeXt-Tiny feature weights and a fresh DSAP-RPB head; any resume restores only this run's own last checkpoint
- ImageNet 权重来源：ConvNeXt_Tiny_Weights.IMAGENET1K_V1
- ImageNet 权重 SHA-256：983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d
- Stage4 smoke 特征形状：[8, 768, 10, 10]
- 参数计数：baseline=27,827,818，DSAP-RPB=28,556,409，新增=728,591
- 训练峰值显存：1.564 GiB
- Smoke Test 峰值显存：1.129 GiB
- 相对位置偏置表理论参数量为 num_heads × (2Hmax−1) × (2Wmax−1)；注意力矩阵复杂度为 O(B × heads × N²)，疾病池化为 O(B × K × N × d)，N=H×W。
- Best checkpoint：\\192.168.60.1\团队文件-Public\multi-label-chest-disease-classification\ConvNeXtTiny\outputs\experiments\ConvNeXt_DSAP_RPB_Cosine20_NoEarlyStop\best.pth
- Last checkpoint：\\192.168.60.1\团队文件-Public\multi-label-chest-disease-classification\ConvNeXtTiny\outputs\experiments\ConvNeXt_DSAP_RPB_Cosine20_NoEarlyStop\last.pth
- Epoch 20 checkpoint：\\192.168.60.1\团队文件-Public\multi-label-chest-disease-classification\ConvNeXtTiny\outputs\experiments\ConvNeXt_DSAP_RPB_Cosine20_NoEarlyStop\epoch20.pth
- 其他产物：config.json、train_log.csv、per_class_metrics.csv、loss_curve.png、auroc_curve.png、per_class_auroc_curve.png、lr_curve.png。
- Attention Map 是模型权重的可视化，不代表经过医学验证的病灶定位。

## 历史 GAP 参照

历史 ConvNeXt GAP best Macro-AUROC=0.791286，Best Epoch=5。
该参照使用的 Backbone LR 与本实验不同，差值不能单独归因于 DSAP-RPB 结构。

## 上一轮 DSAP-RPB 调度实验对照

- 上一轮：best Epoch 7，Macro-AUROC=0.785822，last Epoch=14。
- 本轮：best Epoch 7，Macro-AUROC=0.787977，完成 Epoch=20。
- 本轮相对差值：+0.002155 Macro-AUROC。
- 上一轮在 14 Epoch 处中止；本轮目标为完整 40 Epoch。因此总训练步数不同，不能把差异全部归因于 cosine 周期或低学习率阶段。

## 分阶段趋势

| 阶段 | Epoch 数 | Train Loss 首→末 | Val Loss 首→末 | AUROC 首→末 | 阶段最佳 AUROC |
|---|---:|---:|---:|---:|---:|
| A (1-7) | 7 | 0.44138 → 0.28921 | 0.35818 → 0.32802 | 0.68332 → 0.78798 | 0.787977 (Epoch 7) |
| B (8-20) | 13 | 0.28080 → 0.21194 | 0.34497 → 0.40103 | 0.77666 → 0.76576 | 0.783398 (Epoch 10) |
