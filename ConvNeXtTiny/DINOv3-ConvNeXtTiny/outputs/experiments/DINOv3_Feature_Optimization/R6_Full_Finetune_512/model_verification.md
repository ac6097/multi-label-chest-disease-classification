# R6 全量微调模型核验

- 初始化：`facebook/dinov3-convnext-tiny-pretrain-lvd1689m` revision `10d30274b4d445111e2d5bf75ac93bbd94db274b`，SHA-256 `bd30a9459d6149564ef53af6e8a1999980953b009b94cde836ac1bac4d339cb2`。
- Global Head 按 seed=42 独立初始化，初始 SHA-256 `40dc9552892097293310d9fca16e61913c5fa11eae96e3d56e572553141827a9`，与 R5 初始分类头一致；没有加载 R5/R3 监督微调权重。
- 可训练范围：Stem、Stage1、Stage2、Stage3、Stage4、Final Norm、Global Head；冻结参数为 0。
- Total / trainable / frozen parameters：27,827,818 / 27,827,818 / 0。
- Optimizer groups：`[{"name": "backbone", "parameter_count": 27820128}, {"name": "head", "parameter_count": 7690}]`。
- R5 对齐训练配置：512×512、AdamW、Study BCE、seed 42、相同 LR/weight decay/warmup+cosine/gradient clip/BF16/Study sampler/augmentation/三条件早停；R6 最大 epoch=20。
- Runtime：image budget=8、accumulation=4、channels-last=True、gradient checkpointing=False。
- 全量微调真实 Forward/Backward/Optimizer Step preflight：通过；31 images / 17 Studies / 4 microbatches；Study mean 最大误差 2.38e-07。
- 每个可训练模块均有非零梯度且在 disposable preflight optimizer step 后参数发生变化：`True`。正式训练重新从原始 DINOv3 权重初始化。
- Best Epoch=6，Macro-AUROC=0.798802；停止原因 `three_condition_early_stop`。
- 训练和评估只读取官方 Train / Validation；Test 未读取。
