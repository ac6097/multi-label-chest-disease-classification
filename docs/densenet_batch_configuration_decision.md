# DenseNet-121 Batch 配置决策

## 三种配置

| 用途 | image_budget | grad_accumulation | DataLoader | 实测吞吐 | optimizer updates/epoch |
|---|---:|---:|---|---:|---:|
| A：原始基线 | 16 | 2 | 4 workers，非持久，prefetch 默认 2 | 24.54 images/s，14.97 studies/s | 977 |
| B：最高吞吐候选 | 32 | 1 | 8 workers，persistent，prefetch 2 | 26.74 images/s，16.33 studies/s | 963 |
| C：448 精度实验 | 16 | 2 | 4 workers，非持久，prefetch 默认 2 | 25.94 images/s，15.82 studies/s（统一 8-worker 性能矩阵） | 977 |

配置 C 的吞吐数值来自 batch 对照矩阵的 8-worker 环境；实际 448 训练刻意沿用历史 320 的 4-worker 设置，以保证两次分辨率实验只有输入尺寸不同。

## 取舍

- 32/1 相比 16/2 的吞吐只提高约 3.1%；在约 1,000 张图的样本中两者都做了 33 次 optimizer update，每千张图约 31.7–32.2 次更新。两者完整 epoch 更新数也接近（963 与 977）。
- 但 32/1 的 peak allocated/reserved 从 2.20/2.48 GiB 增至 4.20/4.81 GiB，Study 分组、每次 forward 的 BN 统计和梯度窗口组成也改变。它适合单独工程吞吐实验，不能仅凭速度推断 AUROC。
- 48/1 吞吐低于 16/2，完整 epoch optimizer updates 减少约 35%；64/1 吞吐低约 38%，完整 epoch updates 减少约 51%，且 peak allocated/reserved 超过显卡标称容量。不用于正式精度实验。
- 8-worker persistent 的 steady-state 吞吐只比当前 4-worker 高约 5%，但 worker 启动约 79 秒、主机 RAM 占用更高；4-worker persistent 和 prefetch 4 均未带来吞吐收益。正式匹配实验沿用 320 现有 4 workers。

## 最终选择

- 精度对照继续使用 image_budget=16、grad_accumulation=2，保持 320 匹配对照、原始基线的有效 batch 和更新频率。
- 最高吞吐候选单独记录为 32/1 + 8 workers + persistent_workers + prefetch_factor=2。只有约 3% batch 吞吐提升，且 RAM/启动代价更高；不把它用于 448 消融。
- benchmark 每组使用相同 320 输入、BF16、增强、Study 顺序 seed 42、Epoch 6 初始化及独立模型/optimizer；测试集未参与。

全量 CUDA Events、Study 计数、启动时间及资源采样见 [GPU benchmark 报告](densenet_gpu_benchmark.md)。
