# DenseNet-121 Stage 4 决策

## 决策

保留原始 Epoch 6 320×320 模型作为当前可靠基线，不用 448 checkpoint 替换。448 best Macro-AUROC 为 0.78688，低于原始 Epoch 6 的 0.78883 和匹配 320 continuation 的 0.78927；两个配对 bootstrap 区间都跨 0。448 提升了 Lung Opacity 与 Pneumothorax，但 Pneumonia 的 best-checkpoint AUROC 没有提升，且多个其他类别下降。Study 级推理时延也增加。

## 十个问题

1. **任务管理器约 3.1GB、PyTorch allocated 约 2.2GB 的原因？** PyTorch 16/2 peak allocated/reserved 为 2.20/2.48GiB；WDDM 的 nvidia-smi 是全 GPU 口径，还计入 CUDA context、保留 allocator block 和其他 GPU 进程，因此不能当成模型激活显存。
2. **当前瓶颈是 GPU 还是数据？** 16/2 下 loader wait 均值约 490ms/batch，decode/convert/stack 约 281ms/image；forward/backward CUDA Events 约 44/82ms（当前 4-worker 配置），GPU 利用率中位数 0%。数据加载和 host 同步是主要瓶颈，计算本身并非持续饱和。
3. **16/2、32/1、48/1、64/1 哪个吞吐最好？** 32/1 最高 26.74 images/s；16/2 为 25.94；48/1 为 24.24；64/1 为 16.01。32/1 仅比 16/2 快约 3.1%。
4. **每秒处理提升多少？** 相对 16/2，32/1 images/s 提升 3.1%、studies/s 提升 3.2%；48/1 与 64/1 均下降。8-worker persistent 相对当前 4-worker steady-state 吞吐只提升约 5.2%，启动和 RAM 成本更高。
5. **大 batch 是否改变更新频率和训练行为？** 16/2 与 32/1 在约 1,000 图样本中都做了 33 次 optimizer update，但完整 epoch 分别约 977 和 963 次；48/1、64/1 降至 638、478 次。Study 组成、BN 统计和梯度窗口也变化，所以速度数据不代表 AUROC。
6. **448 是否改善 Pneumonia、Lung Opacity？** best checkpoint Epoch 2 的 Pneumonia AUROC 0.6846，低于 320 的约 0.6926；Lung Opacity 0.7133，高于匹配 320 的 0.7002。Epoch 5 Pneumonia 到过 0.7023，但该 checkpoint 的整体 Macro-AUROC 较低，不作为最终模型。
7. **448 是否提高整体 Macro-AUROC？** 没有。448 为 0.78688，匹配 320 为 0.78927，Epoch 6 baseline 为 0.78883；448−320 的 Subject bootstrap 95% CI 为 [-0.01293, 0.00655]，跨 0。
8. **448 是否更容易过拟合？** 训练 loss 从 0.31767 降到 0.27468；验证 loss 在 Epoch 2 的 0.32276 最低，随后升到 Epoch 6 的 0.33153；Macro-AUROC 也在 Epoch 2 达峰。与后续泛化退化相符，因此 patience 4 在 Epoch 6 停止。
9. **新模型是否替换 73.15 分基线？** 不替换。整体 Macro-AUROC 较低，弱类收益不一致，Study end-to-end 本地均值比 320 高约 11.3%；模型文件大小相近，但 T4 延迟仍未实测。
10. **下一阶段最值得研究什么？** 先做 Pneumonia/Lung Opacity 的定向错误分组与图像区域/预处理分析，保持 320 基线做验证；工程侧优先减少 SMB 网络读取与重复 JPEG decode 的等待。当前证据不支持自动加入 CBAM、ASL 或其他结构。

## 下一阶段约束

- 以 Epoch 6 320 模型为质量基线，不把 448 或某个单类最高 epoch 当成全局 best。
- 把局部区域/预处理问题作为针对弱类的单变量实验；输入读取缓存与模型精度实验分别记录。
- 不在官方测试集上调参或反复提交。
- T4 的 100ms 约束要在 T4 或官方口径复核；RTX 4060 Laptop 的延迟不能替代 T4 测试。

支持数据见 [GPU benchmark](densenet_gpu_benchmark.md)、[448 训练记录](densenet_448_training_results.md) 和 [320/448 对照](densenet_320_vs_448_comparison.md)。
