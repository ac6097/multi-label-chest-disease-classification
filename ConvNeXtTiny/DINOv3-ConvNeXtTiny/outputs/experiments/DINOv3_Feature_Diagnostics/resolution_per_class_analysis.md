# R0/R1/R2 分辨率逐类别诊断

## 数据完整性与复现

- 官方 Validation：859 Studies / 228 Subjects / 1422 images。
- 四组预测按 `(Subject_id, Study_id)` 与官方 Validation Study 表做一对一键连接；重复、缺失和多余 Study 均为 0。标签顺序固定为十类官方顺序，closed-world 规则及目标标签均与官方 Validation CSV 一致。
- Validation 使用确定性 ResizePad、灰度三通道和 ImageNet normalization；没有随机增强。Study 分数按每张图的 raw logits 取算术平均后 sigmoid。
- R2 Best Checkpoint 重推理：Macro-AUROC 0.802116，保存预测 0.802116，绝对差 0.00000000。
- R3 Best Checkpoint 完整融合重推理：Macro-AUROC 0.804653，保存预测 0.804653，绝对差 0.00000000。

## 逐类 AUROC

| Disease | Pos | Neg | R0 · 320 | R1 · 448 | R2 · 512 | R1−R0 | R2−R1 | R2−R0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Enlarged Cardiomediastinum | 24 | 835 | 0.686327 | 0.695135 | 0.721208 | +0.008807 | +0.026073 | +0.034880 |
| Pneumothorax | 37 | 822 | 0.816680 | 0.851779 | 0.852716 | +0.035099 | +0.000937 | +0.036036 |
| Consolidation | 46 | 813 | 0.791339 | 0.789053 | 0.771445 | -0.002286 | -0.017608 | -0.019894 |
| Pneumonia | 68 | 791 | 0.725227 | 0.732422 | 0.730256 | +0.007195 | -0.002166 | +0.005029 |
| Edema | 115 | 744 | 0.871476 | 0.879500 | 0.881972 | +0.008024 | +0.002472 | +0.010496 |
| Cardiomegaly | 177 | 682 | 0.771145 | 0.772620 | 0.779922 | +0.001475 | +0.007302 | +0.008777 |
| Atelectasis | 193 | 666 | 0.774016 | 0.775238 | 0.783862 | +0.001221 | +0.008624 | +0.009845 |
| Lung Opacity | 214 | 645 | 0.729465 | 0.724683 | 0.721760 | -0.004782 | -0.002923 | -0.007705 |
| Pleural Effusion | 260 | 599 | 0.908993 | 0.908546 | 0.907442 | -0.000446 | -0.001104 | -0.001551 |
| No Finding | 306 | 553 | 0.874156 | 0.864205 | 0.870581 | -0.009952 | +0.006376 | -0.003575 |

## 逐类 AUPRC

| Disease | R0 · 320 | R1 · 448 | R2 · 512 | R1−R0 | R2−R1 | R2−R0 |
|---|---:|---:|---:|---:|---:|---:|
| Enlarged Cardiomediastinum | 0.058090 | 0.062480 | 0.073721 | +0.004390 | +0.011241 | +0.015631 |
| Pneumothorax | 0.235324 | 0.319720 | 0.322769 | +0.084397 | +0.003049 | +0.087446 |
| Consolidation | 0.190551 | 0.170732 | 0.158692 | -0.019819 | -0.012040 | -0.031859 |
| Pneumonia | 0.221369 | 0.217121 | 0.220685 | -0.004248 | +0.003564 | -0.000683 |
| Edema | 0.444818 | 0.462462 | 0.476796 | +0.017644 | +0.014334 | +0.031978 |
| Cardiomegaly | 0.427259 | 0.441559 | 0.445410 | +0.014300 | +0.003851 | +0.018151 |
| Atelectasis | 0.471826 | 0.439737 | 0.477368 | -0.032089 | +0.037631 | +0.005542 |
| Lung Opacity | 0.479394 | 0.467609 | 0.432284 | -0.011784 | -0.035325 | -0.047109 |
| Pleural Effusion | 0.790252 | 0.779738 | 0.781878 | -0.010514 | +0.002141 | -0.008374 |
| No Finding | 0.833270 | 0.815526 | 0.822457 | -0.017744 | +0.006931 | -0.010813 |

## Subject-cluster paired Bootstrap

- 2000 paired resamples; Seed=42; unit=Subject. Every sampled Subject contributes all its Validation Studies. The same resampling-index sequence was reused for all model pairs.
- Each replicate skips a class AUROC if it has only one label; macro delta is the mean of available paired class deltas. Per-class confidence intervals use valid replicates only. Intervals are unadjusted for multiple comparisons.

| Comparison | Measure | Δ point | 95% CI | Valid replicates |
|---|---|---:|---:|---:|
| R1 − R0 | Macro-AUROC | +0.004436 | [-0.014100, +0.021090] | 2000 |
| R1 − R0 | Enlarged Cardiomediastinum AUROC | +0.008807 | [-0.115163, +0.127190] | 2000 |
| R1 − R0 | Pneumothorax AUROC | +0.035099 | [-0.015659, +0.101925] | 2000 |
| R1 − R0 | Consolidation AUROC | -0.002286 | [-0.033856, +0.032503] | 2000 |
| R1 − R0 | Pneumonia AUROC | +0.007195 | [-0.029182, +0.044503] | 2000 |
| R1 − R0 | Edema AUROC | +0.008024 | [-0.003073, +0.021264] | 2000 |
| R1 − R0 | Cardiomegaly AUROC | +0.001475 | [-0.019926, +0.021652] | 2000 |
| R1 − R0 | Atelectasis AUROC | +0.001221 | [-0.016544, +0.020848] | 2000 |
| R1 − R0 | Lung Opacity AUROC | -0.004782 | [-0.026500, +0.013033] | 2000 |
| R1 − R0 | Pleural Effusion AUROC | -0.000446 | [-0.006362, +0.007393] | 2000 |
| R1 − R0 | No Finding AUROC | -0.009952 | [-0.020387, -0.000724] | 2000 |
| R2 − R1 | Macro-AUROC | +0.002798 | [-0.005994, +0.010473] | 2000 |
| R2 − R1 | Enlarged Cardiomediastinum AUROC | +0.026073 | [-0.030798, +0.072750] | 2000 |
| R2 − R1 | Pneumothorax AUROC | +0.000937 | [-0.033354, +0.031200] | 2000 |
| R2 − R1 | Consolidation AUROC | -0.017608 | [-0.052130, +0.001668] | 2000 |
| R2 − R1 | Pneumonia AUROC | -0.002166 | [-0.037009, +0.031231] | 2000 |
| R2 − R1 | Edema AUROC | +0.002472 | [-0.005941, +0.012609] | 2000 |
| R2 − R1 | Cardiomegaly AUROC | +0.007302 | [-0.006209, +0.022677] | 2000 |
| R2 − R1 | Atelectasis AUROC | +0.008624 | [-0.009670, +0.024454] | 2000 |
| R2 − R1 | Lung Opacity AUROC | -0.002923 | [-0.021625, +0.017511] | 2000 |
| R2 − R1 | Pleural Effusion AUROC | -0.001104 | [-0.006734, +0.004078] | 2000 |
| R2 − R1 | No Finding AUROC | +0.006376 | [-0.002467, +0.017356] | 2000 |
| R2 − R0 | Macro-AUROC | +0.007234 | [-0.010093, +0.021461] | 2000 |
| R2 − R0 | Enlarged Cardiomediastinum AUROC | +0.034880 | [-0.076804, +0.127628] | 2000 |
| R2 − R0 | Pneumothorax AUROC | +0.036036 | [-0.006651, +0.080029] | 2000 |
| R2 − R0 | Consolidation AUROC | -0.019894 | [-0.056418, +0.010996] | 2000 |
| R2 − R0 | Pneumonia AUROC | +0.005029 | [-0.038739, +0.050102] | 2000 |
| R2 − R0 | Edema AUROC | +0.010496 | [-0.003658, +0.027891] | 2000 |
| R2 − R0 | Cardiomegaly AUROC | +0.008777 | [-0.009489, +0.027369] | 2000 |
| R2 − R0 | Atelectasis AUROC | +0.009845 | [-0.010434, +0.030854] | 2000 |
| R2 − R0 | Lung Opacity AUROC | -0.007705 | [-0.027790, +0.013173] | 2000 |
| R2 − R0 | Pleural Effusion AUROC | -0.001551 | [-0.007613, +0.006362] | 2000 |
| R2 − R0 | No Finding AUROC | -0.003575 | [-0.012992, +0.006376] | 2000 |

## 训练预算与吞吐混杂

| Run | Resolution | Best epoch | Image budget | Accumulation | Effective images/Studies per update | Updates in best epoch | Cumulative updates through best | Checkpointing | Channels-last | Mean images/s through best |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|
| R0 | 320 | 5 | 32 | 1 | 32.0 / 19.5 | 964 | 4816 | True | True | 23.81 |
| R1 | 448 | 6 | 12 | 3 | 35.0 / 21.0 | 878 | 5259 | False | False | 21.86 |
| R2 | 512 | 6 | 12 | 3 | 35.0 / 21.0 | 878 | 5259 | True | False | 22.99 |
| R3 | 512 | 6 | 8 | 4 | 30.5 / 18.5 | 1005 | 6037 | False | True | 22.08 |
- 训练日志中的 `optimizer_updates` 为单轮更新数；上表分别列出 Best Epoch 更新数和截至 Best Epoch 的累计更新数。各组均使用 whole-Study batch sampler，不拆分 Study 或重复/丢弃 Study。

## 观察与限制

- R2−R0 AUROC 最大的点估计类别：Pneumothorax (+0.0360), Enlarged Cardiomediastinum (+0.0349), Edema (+0.0105).
- R2 的单类 AUROC 点估计高于 R0 的类别为 6/10；类别数并不等同于统计支持。请结合上表各类别置信区间判断。
- Macro-AUROC 点估计随 320→448→512 单调上升；是否在 448 后趋平需结合 R2−R1 区间，不能只凭点估计确认平台。
- R0/R1/R2 均使用 P2 冻结策略（Stem/Stage1/Stage2 冻结，Stage3/Stage4/末端 norm 与 Global 分类头训练）；但 R0 的 image budget、梯度累积和累计 optimizer updates 与 R1/R2 不同，R1 与 R2 的梯度 checkpointing 配置也不同。R0→R1/R2 的差异不能完全归因于分辨率。
- 验证集 Best Epoch 参与模型选择；置信区间是条件于该选择的探索性区间，没有校正十类与多组比较，也不是独立测试集确认。

## 图表

- `resolution_auroc_curves.png`：十类 AUROC 曲线与 Macro-AUROC 对比。
- `resolution_gain_by_disease.png`：R2−R0 AUROC 点差与每类阳性/阴性 Study 数。
