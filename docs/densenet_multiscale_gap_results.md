# DenseNet-121 320 输入多尺度 GAP 实验

## 实验问题与设置

检验 Dense Block2/3/4 的 GAP 拼接分类头能否提升 Epoch 6 基线的 Study 级 Macro-AUROC。固定 DenseNet-121 backbone、320×320 输入、Logit Mean 和 Standard BCE；唯一模型结构变量是分类头：

`GAP(Block2) ⊕ GAP(Block3) ⊕ GAP(Block4) → Linear(2560, 10)`

特征宽度为 `512 + 1024 + 1024 = 2560`。未启用 CBAM、Label Cross-Attention 或 Weighted BCE。

| 设置 | 值 |
|---|---|
| 初始化 | Epoch 6 best 的 DenseNet backbone；多尺度分类头按 seed 42 新初始化 |
| 初始化校验 | backbone key 集合及 tensor shape 逐项相同，`strict=True` 加载 |
| 初始化 checkpoint SHA-256 | `8162986883188d168b013ad926f70eccddd559f61d9c922b05ab5db9eff7d477` |
| Optimizer / Scheduler | 新建 AdamW；warmup 2 epoch 后 cosine |
| Backbone / head LR | `5e-6 / 5e-5` |
| Weight decay / gradient clip | `1e-4 / 1.0` |
| 训练预算 | 最多 8 epoch，patience 4；实际完成 8 epoch |
| Image budget / accumulation | `16 / 2` |
| Seed / AMP / workers | `42 / BF16 / 4` |
| 数据 | `cxr_first/train.csv`（18,513 Study）与 `cxr_first/valid.csv`；验证 859 Study、1,422 张图 |
| 设备 | NVIDIA GeForce RTX 4060 Laptop GPU，8 GB |
| 输出 best checkpoint SHA-256 | `A08FFFDDEB5529BADB50B99B86F1E4C4181C40D71F9FBA358BD836704570223A` |

训练命令：

```powershell
F:\Anaconda\envs\cxr\python.exe DenseNet121\train.py `
  --train-config DenseNet121\configs\experiments\multiscale_gap_320_train.yaml `
  --model-config DenseNet121\configs\experiments\multiscale_gap_320_model.yaml `
  --init-backbone-checkpoint DenseNet121\checkpoint\best\best_by_macro_auc.pth
```

源码版本记录：Git HEAD `e5f3ea23250d21c13ecaf85815cd504e136483f9`；本次训练入口 SHA-256 `02356213545098BD047D2B3E030E387E796AE2BCBAB05A7D907AAD4A0D0E3328`。训练配置 SHA-256 `6539AF5F77B8AA16E8E8B4D5889A17500C13973C1194A6B2FDD52AEB0E4F056E`；模型配置 SHA-256 `749F1F9B8D86C13CF6D60634CCA363FD059C45902614731E01B2A1DA090B9816`。

## 训练过程

| Epoch | Train loss | Validation loss | Macro-AUROC | Macro-F1@0.5 | Train 秒 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.36425 | 0.34160 | 0.72619 | 0.12656 | 1423.1 |
| 2 | 0.30427 | 0.32655 | 0.75857 | 0.23916 | 1406.5 |
| 3 | 0.28666 | 0.32931 | 0.76713 | 0.27282 | 1289.2 |
| 4 | 0.27576 | 0.33158 | 0.76777 | 0.27561 | 1278.8 |
| 5 | 0.27072 | 0.34556 | 0.76823 | 0.27797 | 1284.0 |
| 6 | 0.26301 | 0.34789 | 0.76979 | 0.27466 | 1270.7 |
| **7** | **0.26415** | **0.34395** | **0.77040** | **0.27376** | **1262.8** |
| 8 | 0.26143 | 0.34747 | 0.76879 | 0.27709 | 1261.7 |

最佳 epoch 为 7，epoch 8 未刷新。训练累计 `10,476.8` 秒，验证累计 `595.1` 秒；训练日志记录的 PyTorch 峰值分配显存为 `2.207 GB`。训练 checkpoint、日志和验证预测均保存在 `DenseNet121/outputs/experiments/multiscale_gap_320_e6/`。

## 与 Epoch 6 基线对照

基线 fresh validation Macro-AUROC 为 `0.788776`；多尺度 best 为 `0.770401`，差值 **-0.018375**。单图与多图 Study 均低于基线：

| 分组 | Study 数 | Epoch 6 GAP | 多尺度 GAP | 差值 |
|---|---:|---:|---:|---:|
| 全部 | 859 | 0.7888 | 0.7704 | -0.0184 |
| 单图 | 391 | 0.7106 | 0.6952 | -0.0154 |
| 多图 | 468 | 0.8345 | 0.8156 | -0.0188 |

| 类别 | Epoch 6 AUROC | 多尺度 AUROC | 差值 | 多尺度 AUPRC | F1@0.5 |
|---|---:|---:|---:|---:|---:|
| Enlarged Cardiomediastinum | 0.7512 | 0.6943 | -0.0569 | 0.0825 | 0.0000 |
| Pneumothorax | 0.8299 | 0.7748 | -0.0552 | 0.2490 | 0.0000 |
| Consolidation | 0.7531 | 0.7543 | +0.0013 | 0.2055 | 0.0000 |
| Pneumonia | 0.6921 | 0.6845 | -0.0076 | 0.1750 | 0.0000 |
| Edema | 0.8646 | 0.8594 | -0.0052 | 0.4182 | 0.3923 |
| Cardiomegaly | 0.7845 | 0.7774 | -0.0071 | 0.4519 | 0.3650 |
| Atelectasis | 0.7636 | 0.7546 | -0.0090 | 0.4311 | 0.3310 |
| Lung Opacity | 0.6986 | 0.6790 | -0.0196 | 0.4052 | 0.2349 |
| Pleural Effusion | 0.8909 | 0.8802 | -0.0106 | 0.7679 | 0.7143 |
| No Finding | 0.8593 | 0.8454 | -0.0139 | 0.7695 | 0.7000 |

除 Consolidation 小幅提高外，其余类别 AUROC 均下降；Enlarged Cardiomediastinum 与 Pneumothorax 的降幅最大。F1 使用固定阈值 0.5，仅作分数行为记录，不作为竞赛主指标。这个单次 seed/验证划分实验没有证明多尺度 GAP 改善宏平均排序能力；未对这次明显负向差值另做 bootstrap。

## 模型大小与延迟

多尺度 best checkpoint 为 `28,525,329` bytes（约 `28.53 MB`），Epoch 6 基线为 `28,463,825` bytes；增加 `61,504` bytes，远低于 500 MB 上限。

项目基准脚本在验证集抽取 100 个 Study（另有 10 个 warmup），于本机 RTX 4060 Laptop GPU 测得：

| 指标 | Mean | P50 | P95 |
|---|---:|---:|---:|
| 图片解码与预处理 / Study | 145.95 ms | 121.40 ms | 214.82 ms |
| Host-to-device / Study | 1.80 ms | 1.85 ms | 2.71 ms |
| 模型前向与融合 / Study | 41.46 ms | 35.99 ms | 65.80 ms |

模型前向耗时折算平均为 `25.43 ms / 输入图像`。这组延迟来自 4060、本地单 Study 调用，平均每 Study 1.67 张图；不是 T4 实测，也不能据此确认官方 100 ms 口径是否满足。基准 JSON：`DenseNet121/outputs/experiments/multiscale_gap_320_e6/outputs/metrics/latency_4060.json`。

## 结论

本轮没有获得优于 Epoch 6 基线的多尺度 GAP 模型。继续保留 `DenseNet121/checkpoint/best/best_by_macro_auc.pth` 作为当前可靠基线；本实验权重只用于分析，不替换基线。测试集未用于调参或评估。448 分辨率及其他结构实验未启动。
