# ConvNeXt DSAP-RPB 实验摘要

## 实验状态

- 完成轮次：14 / 40
- 停止原因：训练进行中；本文件为当前进度快照
- Best Epoch：7
- Best Validation Macro-AUROC：0.785822
- Best Epoch Train Loss：0.289344
- Best Epoch Val Loss：0.327898
- Best Epoch Train-Val Gap：0.038555
- Best Epoch Macro-F1：0.255797
- Train-Val Loss Gap 定义为 Val Loss − Train Loss。Train 使用增强、Validation 不使用增强，因此 Loss 绝对值并非完全同分布，应结合趋势和 AUROC 判断泛化。

## 网络路径

ConvNeXt-Tiny ImageNet Backbone Stage4 → channel projection → 二维正弦余弦位置编码 → 带二维 Relative Position Bias 的多头空间 Self-Attention → 十个 Disease Query 空间池化 → raw GAP + sigmoid(alpha_k) × disease attention feature → 十个独立疾病分类器。Study 输出继续对图像 Logits 做 Logit Mean。

- Self-Attention 的 Query/Key 来自加入位置编码的 Token，Value 来自投影后的原始 Token；Relative Position Bias 按 (Δh, Δw) 查表并加到 QK logits。
- Position Encoding 是固定二维正弦余弦值，只有缩放参数可训练；Disease Query、Relative Position Bias、投影层和每类 alpha/分类器均参与优化。

## 最佳轮次逐类 AUROC

| 疾病 | AUROC | AUPRC | F1 | 阳性数 | 阴性数 |
|---|---:|---:|---:|---:|---:|
| Enlarged Cardiomediastinum | 0.721158 | 0.075416 | 0.000000 | 24 | 835 |
| Pneumothorax | 0.769646 | 0.168269 | 0.050000 | 37 | 822 |
| Consolidation | 0.757741 | 0.130073 | 0.000000 | 46 | 813 |
| Pneumonia | 0.721090 | 0.178263 | 0.000000 | 68 | 791 |
| Edema | 0.856218 | 0.412105 | 0.198675 | 115 | 744 |
| Cardiomegaly | 0.763341 | 0.416784 | 0.300000 | 177 | 682 |
| Atelectasis | 0.779388 | 0.484647 | 0.295720 | 193 | 666 |
| Lung Opacity | 0.725581 | 0.424905 | 0.277966 | 214 | 645 |
| Pleural Effusion | 0.897611 | 0.780443 | 0.705882 | 260 | 599 |
| No Finding | 0.866444 | 0.809187 | 0.729730 | 306 | 553 |

## 可学习融合系数

- Best checkpoint alpha：Enlarged Cardiomediastinum=0.107331, Pneumothorax=0.109388, Consolidation=0.107691, Pneumonia=0.106174, Edema=0.108124, Cardiomegaly=0.106631, Atelectasis=0.106462, Lung Opacity=0.104304, Pleural Effusion=0.111113, No Finding=0.108416

## 可复现性与产物

- Seed：42
- Optimizer：AdamW，Weight Decay=0.01
- Scheduler：optimizer_update 级 Linear Warmup + Cosine，warmup=2 epochs
- Backbone LR：1e-05；Head LR：0.0002
- 初始化：Original run started from torchvision ImageNet ConvNeXt-Tiny feature weights and a fresh DSAP-RPB head; any resume restores only this run's own last checkpoint
- ImageNet 权重来源：ConvNeXt_Tiny_Weights.IMAGENET1K_V1
- ImageNet 权重 SHA-256：983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d
- Stage4 smoke 特征形状：[8, 768, 10, 10]
- 参数计数：baseline=27,827,818，DSAP-RPB=28,556,409，新增=728,591
- 训练峰值显存：1.561 GiB
- Smoke Test 峰值显存：1.129 GiB
- 相对位置偏置表理论参数量为 num_heads × (2Hmax−1) × (2Wmax−1)；注意力矩阵复杂度为 O(B × heads × N²)，疾病池化为 O(B × K × N × d)，N=H×W。
- Best checkpoint：\\192.168.60.1\团队文件-Public\multi-label-chest-disease-classification\ConvNeXtTiny\outputs\experiments\ConvNeXt_DSAP_RPB\best.pth
- Last checkpoint：\\192.168.60.1\团队文件-Public\multi-label-chest-disease-classification\ConvNeXtTiny\outputs\experiments\ConvNeXt_DSAP_RPB\last.pth
- 其他产物：config.json、train_log.csv、per_class_metrics.csv、loss_curve.png、auroc_curve.png、lr_curve.png。
- Attention Map 是模型权重的可视化，不代表经过医学验证的病灶定位。

## 历史 GAP 参照

历史 ConvNeXt GAP best Macro-AUROC=0.791286，Best Epoch=5。
该参照使用的 Backbone LR 与本实验不同，差值不能单独归因于 DSAP-RPB 结构。
