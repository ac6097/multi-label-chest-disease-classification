# P0/P1/P2 GPU 吞吐量对照

Benchmark 覆盖 NAS 数据加载、图像解码/增强、Forward、Study Logit Mean、BCE、Backward、梯度累积和 AdamW Step。短测筛选候选，入选配置进行 3 次长测；选择满足有效图片数与显存约束的长测均值最高项。

| Profile | Budget | Accum | Workers | Checkpointing | Channels Last | Mean images/s | Std | CV | Effective images/update | Effective studies/update | Peak reserved fraction |
|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|
| P0 | 12 | 3 | 4 | False | True | 34.31 | 0.508 | 0.015 | 35.00 | 21.00 | 0.062 |
| P1 | 32 | 1 | 4 | True | False | 36.73 | 0.190 | 0.005 | 32.00 | 19.50 | 0.114 |
| P2 | 32 | 1 | 4 | True | True | 41.01 | 0.378 | 0.009 | 32.00 | 19.50 | 0.137 |

## 重复测量覆盖

- P0: 3 candidate aggregates; 3 met effective-batch and memory constraints; selected config uses three completed long repeats.
- P1: 3 candidate aggregates; 3 met effective-batch and memory constraints; selected config uses three completed long repeats.
- P2: 3 candidate aggregates; 3 met effective-batch and memory constraints; selected config uses three completed long repeats.

## 参考对照

- D0 历史 Benchmark 均值：47.41 images/s；正式 20 轮训练日志均值约 23.536 images/s。两者计时阶段不同，需分开解释。
- 每个 profile 的候选短测、长测重复和 OOM 记录分别保存在 `p0_gpu_benchmark.csv`、`p1_gpu_benchmark.csv`、`p2_gpu_benchmark.csv`。
- 高于目标有效 Batch 容差或超过显存上限的候选保留在 CSV 中，但不参与最终 runtime 选择。
