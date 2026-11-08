# R11 final report: disease-specific sparse feature routing

## Frozen base and modes

- R11 loads the H2 `best.pth` (Seed 42); every H2 parameter is frozen and the original classifier path is unchanged.
- Base mode (all gates 0) reproduces H2; Full mode enables all ten disease residual branches; Single-Branch-Off and mask ablations are evaluated on the same R11 checkpoint.

## Overall Validation comparison

| Model | Macro-AUROC | Macro-AUPRC | Overall BCE |
|---|---:|---:|---:|
| H2 Base | 0.805038 | 0.424958 | 0.311098 |
| R11 Full | 0.805235 | 0.427619 | 0.308708 |
| Random-Mask (mean over 3 seeds) | 0.804541 ± 0.000238 | 0.425479 ± 0.000711 | 0.309727 |
| Permuted-Mask | 0.805265 | 0.426459 | 0.309436 |

- Base-mode vs cached H2 logits: {"n_studies": 859, "max_abs_logit_diff": 0.0, "label_match": true}.

## Per-class comparison

| Disease | H2 AUROC | R11 AUROC | ΔAUROC | H2 AUPRC | R11 AUPRC | ΔAUPRC | Branch-Off AUROC | Random AUROC (mean) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Enlarged Cardiomediastinum | 0.717440 | 0.716417 | -0.001023 | 0.090365 | 0.089429 | -0.000935 | 0.717440 | 0.717831 |
| Pneumothorax | 0.856530 | 0.856481 | -0.000049 | 0.306424 | 0.305714 | -0.000710 | 0.856530 | 0.854552 |
| Consolidation | 0.793732 | 0.793946 | +0.000214 | 0.197173 | 0.197702 | +0.000528 | 0.793732 | 0.793599 |
| Pneumonia | 0.718181 | 0.718599 | +0.000418 | 0.195093 | 0.195091 | -0.000002 | 0.718181 | 0.714583 |
| Edema | 0.885799 | 0.885227 | -0.000573 | 0.479321 | 0.479347 | +0.000026 | 0.885799 | 0.885566 |
| Cardiomegaly | 0.782792 | 0.782817 | +0.000025 | 0.439611 | 0.439800 | +0.000189 | 0.782792 | 0.783223 |
| Atelectasis | 0.792618 | 0.793641 | +0.001023 | 0.462046 | 0.488986 | +0.026940 | 0.792618 | 0.792396 |
| Lung Opacity | 0.725049 | 0.727103 | +0.002054 | 0.467758 | 0.468876 | +0.001117 | 0.725049 | 0.725458 |
| Pleural Effusion | 0.909574 | 0.909484 | -0.000090 | 0.784624 | 0.783968 | -0.000655 | 0.909574 | 0.909450 |
| No Finding | 0.868666 | 0.868637 | -0.000030 | 0.827164 | 0.827276 | +0.000112 | 0.868666 | 0.868755 |

## Sparse routing diagnostics

- Last epoch channel retention 0.5000; spatial retention 0.5000 (expected 0.5/0.5).
- Same-disease channel IoU 0.9477; spatial IoU 0.4787.
- Cross-disease channel IoU 0.3301; spatial IoU 0.2635.
- Residual |mean| 0.301985; see training history for per-epoch residual statistics.

## Deployment / selector score diagnostics

- Inference throughput (validation, data-loading-bound on the SMB share): 3.88 images/s; peak allocated GPU 0.386 GiB; Best checkpoint 109.2 MB; trainable branch parameters 768,660.
- The training-loop peak memory was not logged by the training driver; the figure above is an inference measurement on the same 8 GB GPU.
- Selector score entropy: channel 5.274, spatial 5.797; per-disease score means/stds/quantiles are in `routing_score_diagnostics.json`.

## Branch ablation

| Disease | Gate ON AUROC | Gate OFF AUROC | Δ (ON−OFF) |
|---|---:|---:|---:|
| Enlarged Cardiomediastinum | 0.716417 | 0.717440 | -0.001023 |
| Pneumothorax | 0.856481 | 0.856530 | -0.000049 |
| Consolidation | 0.793946 | 0.793732 | +0.000214 |
| Pneumonia | 0.718599 | 0.718181 | +0.000418 |
| Edema | 0.885227 | 0.885799 | -0.000573 |
| Cardiomegaly | 0.782817 | 0.782792 | +0.000025 |
| Atelectasis | 0.793641 | 0.792618 | +0.001023 |
| Lung Opacity | 0.727103 | 0.725049 | +0.002054 |
| Pleural Effusion | 0.909484 | 0.909574 | -0.000090 |
| No Finding | 0.868637 | 0.868666 | -0.000030 |

## Answers to the 15 required questions

1. R11 Full Macro-AUROC 0.805235 vs H2 0.805038; Δ=+0.000197, 95% CI [-0.000997, +0.001527]（跨 0，不能视为稳定提升）；AUPRC 0.427619 vs 0.424958, Δ=+0.002661, CI [-0.000494, +0.005266]。
2. AUROC 提高的疾病：Consolidation、Pneumonia、Cardiomegaly、Atelectasis、Lung Opacity。
3. AUROC 下降的疾病：Enlarged Cardiomediastinum、Pneumothorax、Edema、Pleural Effusion、No Finding。
4. 可以精确恢复：Branch-Off 将目标疾病 logit 直接设回 H2 base logit，其余九类保持 R11 residual；目标类 AUROC 与 H2 完全一致（max logit difference 0），其余九类不变。
5. learned mask 是否非随机：Full 0.805235 vs Random 0.804541（Δ +0.000694）；但 Permuted-Mask 0.805265（Δ vs Full +0.000030）。Random 略低、Permuted 与 Full 基本同水平，因此只能认为 learned routing 略优于随机，不能证明疾病专属掩码显著优于“借用其他疾病掩码”。
6. 十个疾病选择差异：跨疾病平均 channel IoU 0.3383、spatial IoU 0.3569；同一疾病跨病例 channel IoU 0.9477、spatial IoU 0.4787（训练诊断最后一轮），说明通道掩码高度样本不变，空间掩码也只有中等样本变化。
7. 贡献最大（ΔAUROC）：Lung Opacity +0.002054、Atelectasis +0.001023、Pneumonia +0.000418。
8. 接近零贡献（|ΔAUROC|<1e-4）：Pneumothorax、Cardiomegaly、Pleural Effusion、No Finding；逐分支 ON/OFF 差异见 per_class_branch_ablation.csv，残差幅值见 sparse_routing_diagnostics.csv。
9. learned vs random：Full 0.805235 vs Random 0.804541（Δ +0.000694）；vs Permuted 0.805265（Δ +0.000030）。差异均在 1e-3 量级，不能据此认定 learned routing 有稳定优势。
10. 正样本 BCE 均值变化：R11−H2 -0.002156；负样本 BCE 均值变化 -0.000118；逐疾病值见 per_class_comparison.csv。
11. 存在过拟合：Best epoch=2（Train BCE 0.231128, Valid BCE 0.308708, AUROC 0.805235）；最后 epoch 5 Train BCE 0.224737 下降但 Valid BCE 0.313964、AUROC 0.803874 变差；三条件早停于 epoch 5。
12. Single-image：H2 0.729328 → R11 0.729535（+0.000207）；Multi-image：H2 0.851752 → R11 0.852286（+0.000534）。两者都只有万分位变化，未见不同趋势。
13. 梯度隔离：preflight 用 Autograd 对十个疾病逐类验证 ∂L_i/∂θ_k=0（i≠k），全部通过；Base 模式与 H2 缓存 logits 最大差 0.0。
14. 新增参数 768,660（约 0.77M）；Best checkpoint 109.2 MB，低于 500 MB；Frozen H2 只保存一份 backbone 权重。
15. Oracle 上限 0.805412（相对最佳单模型 +0.000176）；Full−H2 与 Random−Full 都只有 1e-4~1e-3 量级且 CI 跨 0，因此不建议现在做多随机种子复验；若未来复验，只值得验证 Atelectasis 的 AUPRC（ΔAUPRC +0.026940，CI [+0.001349, +0.052198]）是否可重复。

## Oracle analysis (optimistic upper bound; NOT a formal result)

- Oracle per-class max(H2, R11) Macro-AUROC = 0.805412; Oracle Macro-AUPRC = 0.427849.
- Oracle chooses the better model per disease on Validation; it is post-hoc optimistic and cannot be reported as generalization.

## Limitations

- Single seed; official Validation is reused; all CIs are exploratory.
- No Test labels were read.
- Dense tensor × mask is logical sparsity only; no FLOP reduction is claimed.
- Mask ablations on the same checkpoint cannot isolate the causal effect of training with sparse selection versus a dense unfiltered branch; R7–R10 are historical references only.
- Base-mode numerical equality to H2 is verified against cached H2 logits; BF16 and Study aggregation are unchanged.

