# DINOv3 泛化诊断：环境与 Checkpoint 审计

- 生成时间：2026-11-05
- 审计方式：只读检查项目文件、配置文件、checkpoint 元数据与系统环境；未修改任何既有权重/日志/报告，未读取 Test 标签。
- 结构化原始记录：[environment_and_checkpoint_audit.json](environment_and_checkpoint_audit.json)

## 1. 环境

| 项 | 值 |
|---|---|
| workspace | `ConvNeXtTiny/DINOv3-ConvNeXtTiny/`（公共数据只读引用仓库根 `cxr_first/`） |
| Python / PyTorch | 3.12.3 / 2.5.1（CUDA 12.8 wheel） |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU，8188 MiB |
| cuDNN | 9.1.0；`cudnn.benchmark=False`，`cudnn.deterministic=False` |
| 随机种子 | 训练 seed 42 |
| 依赖 | pandas / numpy / scikit-learn / Pillow / safetensors / PyYAML；无 pytest（测试使用 `unittest`） |

## 2. 数据

| 划分 | Study | 图像 | Subject | 用途 |
|---|---:|---:|---:|---|
| Train | 18,513 | 30,344 | 11,055 | 训练 + G1-A 训练内评价 |
| Validation | 859 | 1,422 | 228 | 官方验证 / 模型选择 / G0 一致性审计 |
| Test | 1,414 | 2,218 | 136 | 仅用于提交推理；**标签未读取** |

- 数据配置：`ConvNeXtTiny/configs/hybrid_data.yaml`（`image_root: ../cxr_first`；`train_csv` / `val_csv`；`image_id_column: Dicom_id`；`subject_column: Subject_id`；`study_column: Study_id`；`label_column: Predict_class`；`label_semantics: closed_world`；`image_size: 320` 由训练/推理运行时覆盖为 512）。
- 标签顺序（官方 ID 1–10）：Enlarged Cardiomediastinum, Pneumothorax, Consolidation, Pneumonia, Edema, Cardiomegaly, Atelectasis, Lung Opacity, Pleural Effusion, No Finding。
- `closed_world`：`Predict_class` 分号分隔阳性类，未列出即阴性；Study 级 target/valid mask 由 `datasets/dataset.py` 构造，全部 10 类 `valid=True`。
- 数据审计来源：`docs/tables/audit_manifest.json`（33,984 张图像，3 个 CSV；图像全部可读，0 读取失败；14 组精确重复文件）。
- 预处理：`ResizePad`（保持长宽比、长边缩放到 512、居中补 0）→ `Grayscale(3)` → `[train] p=0.5 ColorJitter(0.05,0.05) + RandomAffine(translate 2%, scale 0.97–1.03, degrees 0, fill 0)` → `ToTensor` → ImageNet `Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225))`。验证/推理无随机增强。
- R3/R5 训练使用 `MaskedStudyDataset`（`ResizePadWithMask`），额外携带 padding 后有效区域 mask（供 R3 local 分支使用）。

## 3. 预训练权重

- Hugging Face 仓库：`facebook/dinov3-convnext-tiny-pretrain-lvd1689m`
- revision：`10d30274b4d445111e2d5bf75ac93bbd94db274b`
- `model.safetensors` SHA-256：`bd30a9459d6149564ef53af6e8a1999980953b009b94cde836ac1bac4d339cb2`（111,299,216 bytes）
- 本地通过 `huggingface_hub` 缓存严格校验（`local_files_only=True`，hash 匹配才允许构建模型）。R3/R5 checkpoint 的 `initial_weights_sha256` 与之一致。

## 4. Checkpoint

| 实验 | 结构 | Best checkpoint | epoch | best_macro_auroc | state tensor 数 | 文件大小 |
|---|---|---|---:|---:|---:|---:|
| R3 | Stage4 GAP + masked Stage3 Top-K 局部疾病响应（α 融合） | `R3_Global_Local/best.pth` | 6 | 0.804653 | 187 | 324,313,957 B |
| R5 | Global-only GAP + Linear(768,10) | `R5_Global_Only_512/best.pth` | 6 | 0.803514 | 184 | 324,263,357 B |

- R3 checkpoint 恢复的融合系数：`alpha.logit = -1.0810359716`，`alpha = maximum_alpha * sigmoid(alpha.logit) = 0.0506620`（`maximum_alpha=0.2`，初始 α=0.05）。
- R3 结构参数：`backbone`（DINOv3 ConvNeXt-Tiny，27,820,128）+ `classifier` Linear(768,10)（7,690）+ `local_classifier` Conv2d(384,10,1)（3,850）+ `alpha`（1）；总 27,831,669；可训练 26,596,629；冻结 1,235,040（Stem/Stage1/Stage2）。
- R5 结构参数：`backbone` + `classifier` Linear(768,10)；总 27,827,818；可训练 26,592,778；冻结 1,235,040。
- 两者均为 `p2` 冻结配置（Stem、Stage1、Stage2 冻结；Stage3、Stage4、final norm、分类头可训练），均为 seed 42、9/20 epoch 三条件早停。
- 权重文件同时包含 optimizer/history/RNG 等训练状态；推理入口只读取 `model_state`（R3）或独立 `model.safetensors`（提交包）。

## 5. 训练配置（以 R3/R5 实际 config.json 为准）

| 项 | 值 |
|---|---|
| 输入 | 512×512（ResizePad + Grayscale3 + ImageNet normalize） |
| Backbone | DINOv3 ConvNeXt-Tiny（官方通用视觉预训练，非医学权重） |
| 冻结 | Stem、Stage1、Stage2（1,235,040 参数）；Stage3/Stage4/final norm/head 可训练 |
| 优化器 | AdamW，fused=True；backbone LR 1e-5，head LR 2e-4；min LR 5e-7 / 1e-5；weight decay 0.01；默认 betas（0.9,0.999） |
| 调度 | optimizer-update 级 linear warmup 2 epoch + cosine 到 20-epoch 预算；梯度裁剪 1.0 |
| Batch | image_budget=8（每个 microbatch 最多 8 张图、整 Study 不拆分），梯度累积 4，有效约 30.5 张图/18.5 Study 每 update |
| 精度 | BF16 autocast；channels-last=True；梯度检查点关闭；DataLoader 4 workers，pin_memory，prefetch 2 |
| 损失 | `BCEWithLogitsLoss`（逐 Study 内 10 标签平均，再按 Study 平均；`closed_world` 下无未知标签 mask） |
| Study 聚合 | 原始图像 logits 的算术平均（`logit_mean`），聚合后 sigmoid |
| 早停 | 三条件同时满足：train BCE 连续 3 轮下降、valid BCE 连续 3 轮上升、Macro-AUROC 连续 3 轮不提升（实现为硬编码窗口 4 / 严格比较；config 中 `strict_*`/`comparison_epsilon` 等键实际未被读取） |
| 数据划分 | 官方 Train/Validation；无 Subject 泄漏（Train/Valid Subject 集合不相交，由数据构建断言） |

## 6. 推理入口

### 6.1 研究验证入口（E0-A 使用）

- `dino_experiment.validate(model, valid_data, batch_forward=r3_batch_forward, collate_fn=collate_masked_studies)`：
  - 数据：`MaskedStudyDataset(train=False)` + `EpochTaggedStudyDataset`（无随机增强）；`image_budget=8`、`shuffle=False`、4 workers、`channels_last=True`；
  - 模型：R3 `DINOv3GlobalLocalClassifier`（`model(images, valid_masks)`）或 R5 `DINOv3StudyClassifier`（`model(images)`），`eval()`；
  - 精度：CUDA BF16 autocast；
  - 聚合：`study_logit_mean`（同 Study 全部图像 logits 先平均）→ sigmoid → Study 级指标；
  - 输出：`predictions/val_epoch_NN.csv`、`predictions/val_best.csv`、`val_best.csv`（列为 `Subject_id, Study_id, target_*, prob_*`）。

### 6.2 实际提交入口（E0-B 使用）

- R3 提交包：`R3_Global_Local/submission_export/R3_Inference/`
  - `model.safetensors`（111,352,548 B，FP32 推理权重，strict 加载）；
  - `model_config.json`：`head_type=global_local`，`image_size=512`，`channels_last=false`（与训练 runtime 的 `channels_last=true` **不同**），`r3={topk_ratio 0.1, initial_alpha 0.05, maximum_alpha 0.2, mask_valid_fraction_threshold 0.5}`；
  - `dino_feature_inference.FeaturePredictor`：逐 Study 读取图像 → `ResizePadWithMask(512, train=False)` → 单 Study batch → BF16 autocast → `logits.float().mean(dim=0)` → sigmoid；`eval()`，strict 加载；类别顺序来自 `model_config.json`；
  - 提交包内 `dino_model.py`、`dino_feature_models.py`、`dino_feature_inference.py`、`dinov3_meta_convnext.py`、`dino_data.py`、`dino_losses.py`、`datasets/*`、`hybrid/common.py` 与当前仓库同路径文件 **SHA-256 全部一致**（无代码版本漂移）。
- R5 提交包：`R5_Global_Only_512/submission_export/R5_Inference/`（`head_type=gap`，`channels_last=true`）；另有 `global_only_inference.pth` + `global_only_inference_config.json`（从训练 checkpoint 过滤出的推理权重，strict 校验通过）。
- 历史提交文件：`R3_Global_Local/submission.csv`（14,140 行 = 1,414 Test Study × 10 类，文件时间 2026-11-02 22:46）；项目代码中未找到生成该文件的可执行脚本，provenance 不明确。官方 Test 成绩 0.7510 由任务方提供，未在本地复现。

## 7. 已知一致性异常（进入 G0 调查）

- R5 研究推理与历史保存的 `val_best.csv` 不一致：重放最大概率绝对差 0.009352，Macro-AUROC 0.803514（保存）→ 0.803469（重放），此前记录 `cause_resolved=false`（见 `R7_ResidualTransfer_512/validation_replay_and_branch_summary.json`）。
- R7 的同类重放最大概率差为 7.2e-08（一致），说明该现象并非所有运行的普遍行为，需要精度/批次形状/运行间 GPU 数值差异的定位实验。
- 提交包 `channels_last` 与训练 runtime 不同（R3：false vs true），属于需要在 G0 中量化的推理配置差异。

## 8. 本次诊断不修改的内容

- 不修改/覆盖 R3–R10 的 `best.pth`、`last.pth`、训练日志、config、既有报告；
- 不执行 G2 训练，不启动 R11，不引入 L2-SP/SAM/新增强/新注意力；
- 不读取 Test 真实标签，不根据官方 0.7510 做超参数搜索；
- 不改动正式提交方式（Study Logit Mean）。

## 9. 审计结论（进入 G0 前）

1. R3/R5 的模型、冻结范围、优化器/调度/早停实现、Checkpoint 与预训练权重来源均可在仓库中核实，且与训练时记录一致。
2. 提交包与当前研究代码同源（文件哈希一致），因此 G0 主要检查 **推理配置（channels_last/批次形状/BF16）与接口差异**，而非代码版本差异。
3. 存在两个已知待解释项：R5 重放数值差异、R3 提交包与训练 runtime 的 `channels_last` 差异；另需在 G0 中判断 R3 的 Validation 0.804653 能否合理复现。
4. R3 官方 Test 0.7510 与 Validation 0.804653 的落差无法仅凭审计解释；G0/G1 将分别排除推理实现问题和量化 Train/Valid 泛化差距。
