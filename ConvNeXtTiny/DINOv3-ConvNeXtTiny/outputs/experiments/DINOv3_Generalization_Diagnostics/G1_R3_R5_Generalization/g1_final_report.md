# G1 final report: R3/R5 generalization and pretrained-representation drift

Deterministic no-augmentation inference on the official Train (18,513 Studies / 30,344 images / 11,055 Subjects) and Validation (859 Studies / 1,422 images / 228 Subjects) splits. Study Logit Mean before sigmoid, BF16 autocast, channels-last, image budget 8.

## G1-A Train vs Validation (same protocol)

| Group | Split | Macro-AUROC | Macro-AUPRC | Study BCE |
|---|---|---:|---:|---:|
| R3 | train | 0.903671 | 0.621196 | 0.233420 |
| R3 | valid | 0.804516 | 0.418381 | 0.312074 |
| R5 | train | 0.903096 | 0.619838 | 0.236135 |
| R5 | valid | 0.803469 | 0.428405 | 0.311185 |

- Train−Validation Macro-AUROC gap: R3 = +0.099156; R5 = +0.099627.
- Train−Validation Study BCE gap: R3 = -0.078654; R5 = -0.075051.

Largest per-class Train−Valid AUROC gaps (R3 / R5):

| Disease | R3 train | R3 valid | R3 gap | R5 train | R5 valid | R5 gap |
|---|---:|---:|---:|---:|---:|---:|
| Pneumonia | 0.8642 | 0.7228 | +0.1414 | 0.8656 | 0.7242 | +0.1414 |
| Cardiomegaly | 0.9036 | 0.7777 | +0.1259 | 0.9026 | 0.7769 | +0.1257 |
| Consolidation | 0.8878 | 0.7666 | +0.1212 | 0.8872 | 0.7885 | +0.0987 |
| Lung Opacity | 0.8479 | 0.7275 | +0.1204 | 0.8460 | 0.7233 | +0.1227 |
| Atelectasis | 0.8929 | 0.7951 | +0.0978 | 0.8923 | 0.7926 | +0.0997 |
| Enlarged Cardiomediastinum | 0.8389 | 0.7430 | +0.0959 | 0.8388 | 0.7079 | +0.1309 |

## G1-B Single vs multi-image Studies (Validation)

| Group | Study group | Studies | Subjects | Macro-AUROC | Macro-AUPRC | Study BCE |
|---|---|---:|---:|---:|---:|---:|
| R3 | single | 391 | 94 | 0.7282 | 0.3916 | 0.3961 |
| R3 | multi | 468 | 198 | 0.8505 | 0.4381 | 0.2418 |
| R3 | two | 380 | 176 | 0.8563 | 0.4503 | 0.2384 |
| R3 | three_plus | 88 | 60 | 0.7970 | 0.4417 | 0.2565 |
| R5 | single | 391 | 94 | 0.7279 | 0.4025 | 0.3924 |
| R5 | multi | 468 | 198 | 0.8500 | 0.4476 | 0.2434 |
| R5 | two | 380 | 176 | 0.8540 | 0.4608 | 0.2411 |
| R5 | three_plus | 88 | 60 | 0.8082 | 0.4430 | 0.2531 |

Aggregation check (R5 Best, multi-image Validation Studies only; pre-specified Logit Mean vs Probability Mean, no trained aggregator):

- Studies = 468.
- Logit Mean Macro-AUROC = 0.849957.
- Probability Mean Macro-AUROC = 0.846296.
- Δ(Probability−Logit) = -0.003661, 95% CI [-0.00991220623254776, 0.00034100562334060143].

## G1-D parameter drift (relative to original DINOv3)

Note: `stage0`–`stage3` are ConvNeXt stage indices; research "Stage3" = stage2 (9 blocks, 384 ch) and research "Stage4" = stage3 (3 blocks, 768 ch).

| Group | Stage | Absolute L2 drift | Relative drift |
|---|---|---:|---:|
| R3 | stage0 | 0.0000 | 0.000000 |
| R3 | stage1 | 0.0000 | 0.000000 |
| R3 | stage2 | 2.3018 | 0.008433 |
| R3 | stage3 | 2.3711 | 0.007984 |
| R3 | final_norm | 0.0505 | 0.000536 |
| R5 | stage0 | 0.0000 | 0.000000 |
| R5 | stage1 | 0.0000 | 0.000000 |
| R5 | stage2 | 2.2976 | 0.008417 |
| R5 | stage3 | 2.3778 | 0.008006 |
| R5 | final_norm | 0.0524 | 0.000557 |

- Frozen layers (Stem/Stage1/Stage2) bitwise unchanged: True (124 tensors, max |Δ|=0.000000000000).
- R3 learned fusion alpha = 0.050662003742933955; initial 0.05 (alpha.logit 0 → 0.050662).

## G1-E Feature drift (fixed sampled Studies, seed 42)

`final` = LayerNorm-normalized pooled token that the classifier consumes; Stage3/4 GAP are raw channel means (pre-normalization).

| Group | Split | cos(Stage3 GAP) | cos(Stage4 GAP) | cos(final) | CKA Stage3 | CKA Stage4 | CKA final |
|---|---|---:|---:|---:|---:|---:|---:|
| R3 | train | 0.9982 | -0.0124 | 0.0712 | 0.2987 | 0.1746 | 0.2418 |
| R3 | valid | 0.9982 | -0.0126 | 0.0704 | 0.2017 | 0.1823 | 0.3637 |
| R5 | train | 0.9983 | 0.1361 | 0.2131 | 0.2654 | 0.1671 | 0.2296 |
| R5 | valid | 0.9983 | 0.1340 | 0.2087 | 0.1449 | 0.1686 | 0.3650 |

- R3 vs R5 feature cosine (Validation): Stage3 GAP = 0.9996, Stage4 GAP = -0.5979, final = 0.5723.
- Within-run stability control (same run, Best epoch 6 vs Last epoch 9): R3 stage3=0.9997, stage4=0.6099, final=0.6409; R5 stage3=0.9998, stage4=0.5752, final=0.6468. The raw Stage4 GAP vector is therefore highly sensitive to small parameter changes; feature-cosine magnitudes should be interpreted together with the classifier-input (final) features, CKA and the G1-A performance gap.

## G1-C data distribution checks

- Subject overlap Train↔Validation: 0; Study overlap: 0.
- train: 18513 Studies, 30344 images, 11055 Subjects, 1.621 positive labels/Study on average, 37.36% multi-label Studies; Studies/Subject mean 1.675, median 1.0, max 21.
- valid: 859 Studies, 1422 images, 228 Subjects, 1.676 positive labels/Study on average, 40.51% multi-label Studies; Studies/Subject mean 3.768, median 1.0, max 89.

## Evidence table

| Item | Verdict |
|---|---|
| R3 Train−Valid Macro-AUROC gap | +0.099156 (large) |
| R5 Train−Valid Macro-AUROC gap | +0.099627 (large) |
| Gap direction consistent across R3/R5 | True |
| Multi-image Study degradation | R3: single=0.7282, multi=0.8505；R5: single=0.7279, multi=0.8500（组间疾病构成不同，不能直接判优劣） |
| Probability-Mean aggregation worth further study | 目前无法确认：Δ Macro-AUROC 的 95% CI 跨 0 |
| Stage3/4 parameter drift substantial | R3 ConvNeXt-stage2(=Stage3) relative=0.0084, stage3(=Stage4) relative=0.0080；R5 ConvNeXt-stage2(=Stage3) relative=0.0084, stage3(=Stage4) relative=0.0080 |
| Feature drift consistent between R3/R5 | train: R3 final=0.0712, R5 final=0.2131；valid: R3 final=0.0704, R5 final=0.2087 |
| Frozen layers untouched | True |

## Required G1 answers

- **1. R3、R5 的 Train/Valid Macro-AUROC 差距有多大？** R3 = +0.099156（Train 0.903671 → Valid 0.804516）；R5 = +0.099627（0.903096 → 0.803469）。
- **2. 哪些疾病泛化差距最大？** Pneumonia R3 Δ=+0.1414；Cardiomegaly R3 Δ=+0.1259；Consolidation R3 Δ=+0.1212；Lung Opacity R3 Δ=+0.1204
- **3. 哪些疾病即使在 Train 上也表现较差？** Enlarged Cardiomediastinum train AUROC=0.8389；Lung Opacity train AUROC=0.8479；Pneumonia train AUROC=0.8642；Consolidation train AUROC=0.8878
- **4. 是否主要由多图 Study 导致性能下降？** R3 single Macro-AUROC=0.7282 vs multi=0.8505；R5 single=0.7279 vs multi=0.8500。两组疾病构成不同，不能仅凭此判断聚合算法优劣；多图 Study 固定权重聚合对照：Probability Mean − Logit Mean = -0.003661，95% CI [-0.00991220623254776, 0.00034100562334060143]。
- **5. Stage 3 和 Stage 4 哪部分参数变化更大？** R3: Stage3(ConvNeXt stage2) relative=0.008433, Stage4(ConvNeXt stage3) 0.007984；R5: Stage3 0.008417, Stage4 0.008006。Stage4 相对偏移更大是预期现象（任务相关度更高、无冻结差异时更新幅度更大）。
- **6. 两个模型的特征漂移表现是否一致？** train: R3 cos(Stage4 GAP)=-0.0124, R5=0.1361; R3 cos(final)=0.0712, R5=0.2131; CKA Stage4 R3=0.1746, R5=0.1671；valid: R3 cos(Stage4 GAP)=-0.0126, R5=0.1340; R3 cos(final)=0.0704, R5=0.2087; CKA Stage4 R3=0.1823, R5=0.1686
- **7. 是否有证据支持“微调过程改变了模型的泛化表现”？** 存在明确的 Train−Valid 差距与 Stage3/4 参数/特征漂移，但本诊断不能把泛化差距归因于漂移本身：单 seed、验证集反复用于选模、且 Train 评价是训练内评价。可支持的结论仅是“微调改变了表示且存在泛化差距”，不是因果证据。
- **8. 当前是否值得测试更低 Backbone 学习率？** 建议在用户确认后启动 G2（1e-5 vs 3e-6 配对种子实验）。依据：Train−Valid Macro-AUROC 差距 R3=+0.0992/R5=+0.0996；Stage4 参数相对漂移 R3=0.0080/R5=0.0080；分类器输入(final)特征余弦 R3=0.0704/R5=0.2087（原始 Stage4 GAP 余弦 R3=-0.0126/R5=0.1340）。这些是可检验“微调强度影响泛化”的合理依据，但不构成因果证明。

## Limitations

- Train evaluation is an in-training evaluation, not an independent generalization estimate; the Train−Valid gap cannot be attributed to a single cause.
- No independent Test labels are available; the official Test 0.7510 cannot be decomposed into distribution shift vs model error here.
- Parameter/feature drift measures representation change, not causal degradation; a smaller feature cosine similarity is not by itself evidence of overfitting.

