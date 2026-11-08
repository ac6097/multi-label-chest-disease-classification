# P0/P1/P2 冻结状态与梯度验证

三个 profile 均从相同原始 DINOv3 权重和同一随机种子的新分类头独立构建。使用一个真实 Train Study batch 执行 Forward、Study Logit Mean、BCE、Backward 和一次 AdamW Step。

- 初始分类头 SHA-256：`40dc9552892097293310d9fca16e61913c5fa11eae96e3d56e572553141827a9`（三组一致）。
- 官方初始化 SHA-256：`bd30a9459d6149564ef53af6e8a1999980953b009b94cde836ac1bac4d339cb2`。

| Profile | Trainable Parameters | Frozen Parameters | BCE | Trainable Grad Norm | Frozen Gradients | Frozen Weights Unchanged | Trainable Weights Updated |
|---|---:|---:|---:|---:|---|---|---|
| P0 | 7,690 | 27,820,128 | 0.740324 | 7.054074 | True | True | True |
| P1 | 15,479,818 | 12,348,000 | 0.740324 | 28.452103 | True | True | True |
| P2 | 26,592,778 | 1,235,040 | 0.740324 | 38.331995 | True | True | True |

## 实际可训练模块

- **P0**：`classifier`
  - 冻结并保持 eval：`backbone`
- **P1**：`backbone.model.downsample_layers.3, backbone.model.stages.3, backbone.model.norm, classifier`
  - 冻结并保持 eval：`backbone.model.downsample_layers.0, backbone.model.downsample_layers.1, backbone.model.downsample_layers.2, backbone.model.stages.0, backbone.model.stages.1, backbone.model.stages.2`
- **P2**：`backbone.model.downsample_layers.2, backbone.model.stages.2, backbone.model.downsample_layers.3, backbone.model.stages.3, backbone.model.norm, classifier`
  - 冻结并保持 eval：`backbone.model.downsample_layers.0, backbone.model.downsample_layers.1, backbone.model.stages.0, backbone.model.stages.1`

完整的逐参数 `requires_grad` 和逐模块 `training/eval` 状态保存在 `model_freeze_verification.json`。

验证通过。冻结参数未进入 optimizer、没有梯度且一步更新后完全不变；可训练参数均收到梯度且至少一个参数发生更新。
