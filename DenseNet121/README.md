# DenseNet-121 胸片多标签路线

此目录是独立的 DenseNet-121 实验工程；公共源数据只读引用仓库根目录的 `cxr_first/`。路线保留 torchvision DenseNet-121 的四个 Dense Block、密集连接、标准 GAP 分类头、Dense Block2/3/4 多尺度候选、独立 CBAM 和 10 Query Label Cross-Attention。

## 数据和标签

当前 CSV 是每张图一行，字段为 `Dicom_id`、`Subject_id`、`Study_id` 和 `Predict_class` 等。数据加载器按 `(Subject_id, Study_id)` 形成一个样本，从 `split/Subject_id/Study_id/Dicom_id.jpg` 只读载入该 Study 的全部图像。变长图像通过扁平图像张量和 `study_index` 对齐；每个 Study 在每个 epoch 只产生一次标签损失。ID 均以字符串读取。

`Predict_class` 是以分号分隔的阳性类别 ID。现已确认训练集和验证集里未列出的类别为阴性，`configs/data.yaml` 设置为 `label_semantics: closed_world`；每个非空 Study 标签串会转成完整10类 target 和 valid mask。测试集没有读取标签，不参与权重估计或训练。

## 网络和 Study 融合

标准头为 DenseNet 最终特征图的 GAP → `Linear(1024, 10)`。多尺度候选对 Block2/3/4 分别 GAP，拼接 `512 + 1024 + 1024 = 2560` 维向量。CBAM 仅以独立实验开关接在最终特征图；Label Cross-Attention 使用 10 个可学习 Query、`d_model=256`、4 个头、1 层交叉注意力和 0.1 dropout。CBAM、多尺度头和 Label Cross-Attention 不会由默认配置同时叠加。

`configs/model.yaml` 的 `study_fusion` 支持：

- `logit_mean`：默认 Study 级基线，在 sigmoid 前平均每图 logits。
- `probability_mean`、`probability_max`：先对逐图 logits 做 sigmoid，再在 Study 内聚合概率。
- `feature_mean`、`feature_max`：先聚合 DenseNet GAP 图像向量，再送入分类头。
- `attention_pooling`：以学习到的注意力权重聚合图像向量，需要 Study 级训练。

第一轮只需将同一模型配置的 `study_fusion` 在 `logit_mean` 和 `probability_mean` 间对照。切换骨干、损失、分辨率或融合方式时，保持其他设置和 Subject 划分一致。

## 损失和训练设置

`configs/train.yaml` 支持标准 `bce`、`weighted_bce` 与候选 `asl`。Weighted BCE 可选择 `weighting: raw | sqrt | cap`，截断上限由 `pos_weight_cap` 控制；类别权重只从训练 Study 的有效标签估计。未知标签通过逐元素 mask 排除，损失先按 Study 的有效标签数归一化，再对 Study 平均。ASL 初始配置为 `gamma_pos=0, gamma_neg=4`，仅作为消融候选。

当前建议超参数为 320 输入、图像预算 16、梯度累积 2、AdamW、backbone/head 学习率 `2e-5/2e-4`、weight decay `1e-4`、2 epoch warmup 后 cosine、最多 40 epoch、patience 8、梯度裁剪 1.0。ImageNet 权重使用 torchvision 官方权重。训练/验证已按确认的闭世界语义配置，可在本目录执行 `python train.py` 启动。

DenseNet 默认只在冻结 backbone 的首轮冻结 BN 统计。`freeze_batchnorm_stats` 可单独作为小 batch 稳定性消融开关；比较时不要同时改动 CBAM、多尺度或标签注意力。

## 命令

在 `DenseNet121/` 目录运行：

```bash
python -m pip install -r requirements.txt
python train.py
python evaluate.py --checkpoint checkpoint/best/best_by_macro_auc.pth
python predict.py --checkpoint checkpoint/best/best_by_macro_auc.pth
python benchmark_latency.py --checkpoint checkpoint/best/best_by_macro_auc.pth
```

训练按 E0→E8 逐步消融：Study 数据与标准 BCE 基线、损失对照、多图融合、分辨率、多尺度、CBAM、Label Cross-Attention、仅组合已验证模块、最后分析 DenseNet 与 ConvNeXt 误差互补。每次验证以 Study 为单位记录 Loss、Macro-AUROC、逐类 AUROC/AUPRC/F1、有效阳性/阴性数量、耗时和显存；缺正类或缺负类时 AUROC 为不可计算值。

`predict.py` 生成官方长表 `Subject_id, Study_id, Predict_class, Probability`，每个测试 Study 输出类别 1–10 共 10 行。最终文件可用仓库根目录的 `scripts/validate_submission.py` 校验。`benchmark_latency.py` 在 checkpoint 生成后记录配置中融合方式的 Study 前向 p50/p95，以及图像解码预处理与设备传输时间。T4 实测延迟尚未运行；赛事官网约束为单张图像端到端不超过 100 ms，多图 Study 的总延迟如何计入仍未说明。

`checkpoint/best/best_by_macro_auc.pth` 只保存模型权重和推理配置，供评估/预测使用；`checkpoint/last/last.pth` 保存优化器等训练状态，供断点续训使用。部署提交应选权重文件并按官方方式导出，单个模型文件仍需在 T4 提交包构建时确认不超过 500 MB。
