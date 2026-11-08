# R5 初始化核验

- 官方通用视觉预训练权重：`facebook/dinov3-convnext-tiny-pretrain-lvd1689m` revision `10d30274b4d445111e2d5bf75ac93bbd94db274b`。
- 权重 SHA-256：`bd30a9459d6149564ef53af6e8a1999980953b009b94cde836ac1bac4d339cb2`；通过与 R3 相同的本地缓存和 SHA 校验流程加载。
- Backbone 从原始通用视觉预训练权重初始化，没有加载 R2/R3 胸片监督微调权重。
- Global Head seed=42，构造顺序与 R3 相同；初始 SHA-256 `40dc9552892097293310d9fca16e61913c5fa11eae96e3d56e572553141827a9` 与 R3 记录一致。
- R5 Best Epoch=6，Macro-AUROC=0.803514。
- 权重专用文件：`global_only_inference.pth`；仅包含 Backbone 和 Global Linear(768,10) 参数，不包含 Optimizer/Scheduler 状态。

