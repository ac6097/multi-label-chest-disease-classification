# DenseNet-121 GPU 吞吐量与瓶颈诊断

## 实验范围

- 设备：NVIDIA GeForce RTX 4060 Laptop GPU，8,188 MiB；PyTorch 2.10.0+cu128，CUDA 12.8，BF16。
- 输入：320×320，DenseNet-121 标准 GAP、Standard BCE、Study Logit Mean。
- 初始化 checkpoint：Epoch 6 best；SHA-256 为 8162986883188d168b013ad926f70eccddd559f61d9c922b05ab5db9eff7d477。
- 训练集：18,513 Study、30,344 张图。图像目录解析到 SMB 网络共享路径 \\192.168.60.1\团队文件-Public\multi-label-chest-disease-classification\cxr_first。
- 每个场景独立加载模型和 AdamW；Study 顺序 seed=42；输入使用原训练增强；每场景预热 4 个 batch，测量约 1,024 张图。
- CUDA Events 测量 H2D、forward、loss、backward、optimizer step；另以 500ms 采样 nvidia-smi GPU 利用率，以 Windows 系统计数器采样整机 CPU 利用率。

原始结果和明细见 [benchmark_report.json](../DenseNet121/outputs/experiments/densenet_gpu_benchmark/benchmark_report.json) 与 [benchmark_summary.csv](../DenseNet121/outputs/experiments/densenet_gpu_benchmark/benchmark_summary.csv)。

## DataLoader 对照

所有配置使用 16/2，steady-state 吞吐计时不含 worker 启动时间；启动耗时单独记录。

| 配置 | 启动秒 | images/s | studies/s | loader 等待 ms/batch | 相对当前吞吐 |
|---|---:|---:|---:|---:|---:|
| 当前：4 workers，非持久，prefetch 2 | 29.75 | 24.54 | 14.97 | 489.84 | — |
| 0 workers | 1.60 | 9.62 | 5.87 | 1,473.35 | -60.8% |
| 2 workers，持久，prefetch 2 | 19.38 | 19.53 | 11.91 | 672.83 | -20.4% |
| 4 workers，持久，prefetch 2 | 38.97 | 24.34 | 14.84 | 490.65 | -0.8% |
| 8 workers，持久，prefetch 2 | 79.22 | 25.82 | 15.75 | 452.06 | +5.2% |
| 4 workers，持久，prefetch 4 | 59.24 | 24.03 | 14.65 | 479.94 | -2.1% |

8 个 worker 的 steady-state 吞吐最高，但比当前只快约 5%，启动时间约为当前的 2.7 倍；每个 Windows worker 的实测工作集约 0.9–1.1GB。prefetch 4 未提高吞吐。当前 4 workers 在吞吐和主机内存成本间更合适；分辨率实验也保留现有 4-worker 设置以匹配 320 对照。

## Image budget 与 gradient accumulation 对照

下面四组都使用 benchmark 选出的 8 workers、持久 worker、prefetch 2，以保持候选之间的数据读取配置一致。每组按约 1,024 张实图计时。实际 batch 是完整 Study 打包结果，image_budget 是上限而非固定 batch size。

| image_budget / accumulation | 实测图像 / Study | 平均图像 / Study 每 batch | images/s | studies/s | updates/实测数据 | optimizer updates/epoch | peak allocated / reserved GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| 16 / 2 | 1,025 / 625 | 15.53 / 9.47 | 25.94 | 15.82 | 33 / 1,025 张图 | 977 | 2.20 / 2.48 |
| 32 / 1 | 1,040 / 635 | 31.52 / 19.24 | 26.74 | 16.33 | 33 / 1,040 张图 | 963 | 4.20 / 4.81 |
| 48 / 1 | 1,045 / 636 | 47.50 / 28.91 | 24.24 | 14.75 | 22 / 1,045 张图 | 638 | 6.23 / 7.11 |
| 64 / 1 | 1,071 / 656 | 63.00 / 38.59 | 16.01 | 9.81 | 17 / 1,071 张图 | 478 | 8.28 / 9.39 |

实际 batch 波动较小：16 budget 为 14–16 张图、6–12 个 Study；32 为 30–32 张图、15–23 个 Study；48 为 46–48 张图、25–34 个 Study；64 为 62–64 张图、33–45 个 Study。

| 配置 | H2D ms/batch | forward ms/batch | loss ms/batch | backward ms/batch | optimizer step ms |
|---|---:|---:|---:|---:|---:|
| 16 / 2 | 2.06 | 51.90 | 0.78 | 93.17 | 15.37 |
| 32 / 1 | 5.88 | 115.35 | 0.82 | 157.74 | 14.00 |
| 48 / 1 | 9.49 | 160.36 | 0.81 | 205.10 | 13.66 |
| 64 / 1 | 6.50 | 1,031.24 | 0.73 | 2,855.59 | 39.44 |

64/1 没有抛出 OOM，但 PyTorch peak allocated 达 8.28 GiB、reserved 达 9.39 GiB，均超过设备标称 8 GiB；吞吐比 16/2 低 38%。不能把本次成功分配当作安全余量。

## 瓶颈与显存解释

- 当前 4-worker/16/2 的 decode、RGB convert、stack 约 280.61 ms/image，Resize/增强/归一化约 27.65 ms/image，H2D 约 1.80 ms/batch，loader 等待均值 489.84 ms/batch。同组 CUDA forward/backward 均值约 43.58 / 82.10 ms/batch。当前吞吐主要受 SMB 文件读取、JPEG 解码及 CPU 预处理限制，GPU 不能持续获得 batch。
- 16/2 场景抽样的整机 GPU 利用率均值约 3.3%、中位数 0%、P95 25.7%；CPU 为整机计数器，不是训练进程单独 CPU 使用率。Windows WDDM 的 GPU 利用率在不同场景间波动较大，应与 CUDA Events 和 loader wait 一起看。
- 原训练循环每个 batch 都有 known.sum().item()、float(loss.detach())，每个 optimizer update 还把 gradient norm 转成 Python float。16/2 场景这些显式 host sync 调用合计约 37.5ms/batch；这会妨碍 CPU/GPU 重叠。
- 16/2 与 32/1 的峰值显存不同，原因是 32/1 同时保留约两倍图像的 forward activations；16/2 的两个 microbatch 顺序 forward/backward，前一 microbatch 的 activations 在下一次 forward 前释放。gradient accumulation 本身不把两个 microbatch 的 activation 同时保留。
- 16/2 PyTorch peak allocated/reserved 为 2.20/2.48 GiB；nvidia-smi/任务管理器统计整张 WDDM GPU，还计入 CUDA context、保留 allocator block 和 PixPin、QQ 等其他 GPU 进程。报告中系统 GPU memory 样本中位数约 2.8GiB，不能与 PyTorch allocated 等同。16/2 的 inactive split peak 约 0.21 GiB，没有明显碎片或 OOM 证据。

当前推荐和精度实验选择见 [batch 配置决策](densenet_batch_configuration_decision.md)。
