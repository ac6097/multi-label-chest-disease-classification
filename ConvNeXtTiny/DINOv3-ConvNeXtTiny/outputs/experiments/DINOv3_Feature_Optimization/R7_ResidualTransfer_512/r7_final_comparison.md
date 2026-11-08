# R7 Residual Transfer Final Comparison

- Macro-AUROC: R5 `0.803469`; R7 Fused `0.802901`; R7 gamma=0 `0.802953`.
- Macro-AUPRC: R5 `0.428405`; R7 Fused `0.427874`; R7 gamma=0 `0.427953`.
- R7 Fused−R5 Macro-AUROC: `-0.000568` 95% CI `[-0.001113259792327881, -8.528530597686772e-05]`.
- R7 Fused−R7 gamma=0 Macro-AUROC: `-0.000052` 95% CI `[-0.00015056729316711425, 2.3426115512847892e-05]`.
- R7 Fused−R5 Macro-AUPRC: `-0.000531` 95% CI `[-0.004722967743873596, 0.0009483061730861661]`.
- R7 gamma=0−R5 Macro-AUROC: `-0.000516` 95% CI `[-0.0010370075702667235, -2.2922456264495988e-05]`.
- R7 Best Epoch `6` of 9; stop `three_condition_early_stop`; Best checkpoint gamma `-0.00020749128`, final-stop gamma `-0.00023250337`.
- Best Validation replay max probability error vs saved R7 prediction: `7.17e-08`.
- R5 saved-prediction replay: max `|Δp|=0.009352`; saved/replayed Macro-AUROC `0.803514` / `0.803469`; max per-class AUROC delta `0.000418`. Same Study keys and labels matched; paired comparisons use fresh R5 Best inference from the same Validation loader as R7. The score-level discrepancy's cause remains unresolved.

## Per-disease results

| Disease | Pos/Neg | R5 AUROC | R7 Fused AUROC | R7 gamma=0 AUROC | R7−R5 AUROC Δ (95% CI) | Fused−gamma0 AUROC Δ (95% CI) | R5 AUPRC | R7 Fused AUPRC | R7 gamma=0 AUPRC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Enlarged Cardiomediastinum | 24/835 | 0.7079 | 0.7065 | 0.7066 | -0.0014 [-0.0058, +0.0023] | -0.0001 [-0.0007, +0.0003] | 0.0972 | 0.0968 | 0.0964 |
| Pneumothorax | 37/822 | 0.8552 | 0.8548 | 0.8548 | -0.0004 [-0.0018, +0.0007] | +0.0000 [-0.0001, +0.0001] | 0.3512 | 0.3571 | 0.3577 |
| Consolidation | 46/813 | 0.7885 | 0.7884 | 0.7884 | -0.0000 [-0.0015, +0.0024] | +0.0000 [-0.0005, +0.0004] | 0.1998 | 0.1965 | 0.1969 |
| Pneumonia | 68/791 | 0.7242 | 0.7234 | 0.7236 | -0.0008 [-0.0033, +0.0022] | -0.0001 [-0.0007, +0.0004] | 0.2032 | 0.1988 | 0.1989 |
| Edema | 115/744 | 0.8854 | 0.8851 | 0.8851 | -0.0003 [-0.0009, +0.0003] | -0.0000 [-0.0001, +0.0001] | 0.4782 | 0.4765 | 0.4765 |
| Cardiomegaly | 177/682 | 0.7769 | 0.7763 | 0.7764 | -0.0006 [-0.0020, +0.0008] | -0.0001 [-0.0003, +0.0001] | 0.4309 | 0.4305 | 0.4308 |
| Atelectasis | 193/666 | 0.7926 | 0.7923 | 0.7923 | -0.0003 [-0.0012, +0.0006] | +0.0000 [-0.0001, +0.0002] | 0.4621 | 0.4629 | 0.4627 |
| Lung Opacity | 214/645 | 0.7233 | 0.7223 | 0.7224 | -0.0011 [-0.0021, +0.0000] | -0.0001 [-0.0003, +0.0000] | 0.4574 | 0.4565 | 0.4565 |
| Pleural Effusion | 260/599 | 0.9088 | 0.9082 | 0.9083 | -0.0005 [-0.0010, -0.0001] | -0.0000 [-0.0001, +0.0000] | 0.7768 | 0.7760 | 0.7761 |
| No Finding | 306/553 | 0.8720 | 0.8717 | 0.8718 | -0.0002 [-0.0010, +0.0005] | -0.0001 [-0.0002, +0.0000] | 0.8271 | 0.8271 | 0.8272 |

## Residual branch diagnostics

- Best-epoch gamma: `-0.00020749128`.
- Full Validation image residual ratio mean / median / P90 / max: `0.000654` / `0.000636` / `0.000867` / `0.001585`.
- Stage4-residual cosine mean / median: `0.008313` / `0.009237`.
- Disposable two-step preflight: initial gamma gradient `0.265257`; gamma after first step `0.00019999973`; projection gradient at zero gamma `0`; projection gradient after gamma moved `0.0030317`.
- `r7_branch_diagnostics.csv` records gamma, per-epoch gamma-gradient statistics, fixed Validation-batch residual ratios/cosines, and the Best checkpoint full-Validation residual summary.

## Training and resource use

- R7 trainable/frozen parameters: `26,887,691` / `1,235,040`.
- Added branch parameters: projection `294,912`, gamma `1`.
- Best checkpoint bytes: `327,809,824`.
- Standalone inference archive: `104,652,777` bytes (strict < `500,000,000` passed); model-only safetensors `112,516,724` bytes.
- Per-Study package Validation inference: Macro-AUROC `0.802872`, Macro-AUPRC `0.427731`; mean forward latency `39.32` ms/Study; end-to-end `282.48` ms/Study (`5.86` images/s, including NAS reads and preprocessing).
- Standalone per-Study versus batched Validation replay max `|Δp|=0.006612`, Macro-AUROC `0.802872` vs `0.802901`; batch-shape precision is a plausible contributor, not a proven explanation for the R5 historical replay difference.
- Formal Train throughput through each Best Epoch: R5 `22.14` vs R7 `19.89` images/s (`-10.2%`); cumulative optimizer updates `6037` vs `6037`. Peak reserved training memory: R5 `1.57` vs R7 `1.75` GiB.
- Train/Validation plots compare R5, R6, and R7; the R7 early-stop rule and R5 runtime were reused.

## Interpretation

`R7 Fused−R7 gamma=0` compares two forward paths using the same trained R7 weights and isolates the inference-time direct branch contribution. `R7 gamma=0−R5` compares separately trained models and cannot be attributed causally to the branch because optimization trajectories differ.
The branch can provide a disease-specific effect without improving Macro-AUROC. Validation was used for Best-checkpoint selection and contains prior model comparisons; all paired intervals are exploratory and exposed to multiple-comparison risk. These measurements do not prove that Stage 4 irreversibly loses information.
