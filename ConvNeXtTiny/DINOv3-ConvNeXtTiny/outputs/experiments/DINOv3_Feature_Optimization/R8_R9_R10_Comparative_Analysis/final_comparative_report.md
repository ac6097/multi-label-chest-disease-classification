# R8 / R9 / R10 comparative analysis

All models were independently trained from the same pinned DINOv3 general-vision weights with the R5 P2 freeze profile, R5 selected runtime (image budget 8, gradient accumulation 4, BF16, channels-last), Study Logit Mean BCE, seed 42 and the three-condition early stopping. Only the classifier topology differs.

## Experiment matrix

| Experiment | Auxiliary head | Best epoch | Trainable params | Unified Macro-AUROC | Unified Macro-AUPRC |
|---|---|---:|---:|---:|---:|
| R5 | R5_Global_Only_512 | 6 | 27,827,818 | 0.803469 | 0.428405 |
| R7 | R7_ResidualTransfer_512 | 6 | 28,122,731 | 0.802901 | 0.427874 |
| R8 | R8_DiseaseSpecific_StageLogitFusion_512 | 6 | 27,832,446 | 0.803521 | 0.428488 |
| R9 | R9_DiseaseSpecific_QueryAttention_512 | 6 | 27,929,726 | 0.802963 | 0.427698 |
| R10 | R10_Stage4Guided_CrossStageAttention_512 | 6 | 28,029,694 | 0.803098 | 0.435072 |

## Mechanism findings (best-checkpoint diagnostics)

- R8: best epoch 6，alpha 范围 0.0883–0.1000；Stage3 辅助 logit 残差相对主 logit 的平均比例 0.5796%（最大 1.7264%），辅助支路幅值极小。 Full−BranchOff = +0.000076，95% CI [-0.000043, +0.000185]。
- R9: best epoch 6，归一化注意力熵 1.0000（1.0 表示完全均匀），最大注意力权重均值 0.00098，疾病间注意力平均余弦 1.00000。注意力与均匀分布不可区分（塌缩），没有学到疾病特异的空间选择。 Full−Uniform = ≈0，95% CI [-0.000010, +0.000005]。 Full−BranchOff = -0.000026，95% CI [-0.000139, +0.000089]。
- R10: best epoch 6，Stage4 条件向量与疾病 embedding 的范数比 124:1；归一化注意力熵 0.806，疾病间注意力平均余弦 0.99988。条件项主导了 Query，使十个疾病 Query 与注意力分布几乎相同：注意力是图像自适应的，但疾病特异性被条件项淹没。 Full−NoStage4Conditioning = +0.000713，95% CI [-0.001094, +0.002507]。

## Paired comparison table (fresh unified predictions)

| Comparison | Δ Macro-AUROC | 95% CI | Δ Macro-AUPRC | 95% CI |
|---|---:|---|---:|---|
| R8_Full_minus_R5 | +0.000052 | [-0.000457, +0.000512] | +0.000083 | [-0.002912, +0.001611] |
| R8_Full_minus_R8_BranchOff | +0.000076 | [-0.000043, +0.000185] | +0.000202 | [-0.000499, +0.001096] |
| R8_BranchOff_minus_R5 | -0.000025 | [-0.000513, +0.000431] | -0.000119 | [-0.002974, +0.001355] |
| R9_Full_minus_R5 | -0.000506 | [-0.000972, -0.000018] | -0.000707 | [-0.004972, +0.001103] |
| R9_Full_minus_R8_Full | -0.000558 | [-0.001041, -0.000008] | -0.000790 | [-0.003846, +0.000760] |
| R9_Full_minus_R9_BranchOff | -0.000026 | [-0.000139, +0.000089] | +0.000164 | [-0.000031, +0.000599] |
| R9_Full_minus_R9_Uniform | -0.000000 | [-0.000010, +0.000005] | -0.000001 | [-0.000008, +0.000000] |
| R9_BranchOff_minus_R5 | -0.000480 | [-0.000946, -0.000012] | -0.000870 | [-0.005221, +0.000883] |
| R10_Full_minus_R5 | -0.000371 | [-0.003585, +0.002651] | +0.006667 | [-0.003363, +0.012246] |
| R10_Full_minus_R9_Full | +0.000135 | [-0.003059, +0.003124] | +0.007374 | [-0.002054, +0.012863] |
| R10_Full_minus_R10_BranchOff | +0.000807 | [-0.001073, +0.002668] | +0.001103 | [-0.002624, +0.005223] |
| R10_Full_minus_R10_NoStage4Conditioning | +0.000713 | [-0.001094, +0.002507] | +0.000672 | [-0.003161, +0.004749] |
| R10_BranchOff_minus_R5 | -0.001178 | [-0.004273, +0.001763] | +0.005564 | [-0.005111, +0.010966] |
| R7_Fused_minus_R5 | -0.000568 | [-0.001113, -0.000085] | -0.000531 | [-0.004723, +0.000948] |
| R7_Fused_minus_R7_gamma0 | -0.000052 | [-0.000151, +0.000023] | -0.000080 | [-0.000303, +0.000135] |

## Required questions

- **1. R8 的疾病专属融合是否优于 R5 和 R7？** 目前无法确认方向性差异：Δ=+0.000052 95% CI [-0.000457, +0.000512] 跨 0
- **2. R8 是否成功学习到不同疾病的有效辅助信息？** 目前无法确认方向性差异：Δ=+0.000076 95% CI [-0.000043, +0.000185] 跨 0
- **3. R9 的 Query Attention 是否优于简单 GAP 融合？** 有负向差异，95% CI 排除 0：Δ=-0.000558 [-0.001041, -0.000008]
- **4. R9 的注意力是否表现出疾病差异和图像自适应性？** 注意力塌缩：归一化熵 1.0000–1.0000（1.0 = 均匀分布），平均最大权重 0.00098，疾病间注意力平均余弦 1.00000 → 学到的空间注意力与均匀分布不可区分，未表现出疾病差异或图像自适应。
- **5. R10 的 Stage4 Query Conditioning 是否产生实际贡献？** 目前无法确认方向性差异：Δ=+0.000713 95% CI [-0.001094, +0.002507] 跨 0；条件项主导 Query、疾病间注意力塌缩的机制证据见 Mechanism findings。
- **6. 哪些疾病从 Stage 3 的补充信息中获益？** Enlarged Cardiomediastinum（R8_Full_minus_R8_BranchOff Δ=+0.0008）；Lung Opacity（R10_Full_minus_R10_BranchOff Δ=+0.0023）；Lung Opacity（R10_Full_minus_R10_NoStage4Conditioning Δ=+0.0022）
- **7. 哪些疾病反而受到辅助信息干扰？** 目前无法确认：没有 CI 排除 0 的负向差异
- **8. 三种新结构中，哪一种在性能、复杂度、稳定性之间最有优势？** Macro-AUROC 点估计最高为 R8（0.803521，参数 27,832,446，85.6 ms/Study）；Macro-AUPRC 点估计最高为 R10（0.435072）；参数最少为 R8（27,832,446）。R10 的 AUPRC 点估计相对 R5 提高约 +0.0067，但配对 95% CI 跨 0；因各结构参数、推理成本与置信区间不同，不能宣称某一结构在全部维度上占优。
- **9. 目前的结果是否支持继续优化特征选择，还是应重新考虑 Backbone 特征提取？** 当前证据不支持继续加码特征选择：没有任何新结构相对 R5 的 Macro-AUROC CI 排除 0 且为正，且 R9_Full_minus_R5 的 95% CI 排除 0 但方向为负，辅助分支的实际推理贡献有限。若要继续，应先做独立种子复验；另一方向是回到 Backbone/预训练特征提取本身。
- **10. 是否值得在后续独立研究 Study 级加权 Logit Mean？** 本轮未研究 Study 级加权 Logit Mean（按要求继续使用算术平均），现有数据无法支持或否定该方向，属未来独立实验。

## Interpretation limits

- Validation has been used for repeated checkpoint selection and many exploratory comparisons across R1-R10; all intervals are conditional and exploratory.
- R8/R9/R10 differ in parameter count and head capacity, so a performance difference cannot be attributed solely to attention or conditioning without a capacity-matched control.
- Attention maps and branch statistics describe model behaviour; they are not verified lesion localization.
- Test labels were never read and the Test split was not used for tuning.

