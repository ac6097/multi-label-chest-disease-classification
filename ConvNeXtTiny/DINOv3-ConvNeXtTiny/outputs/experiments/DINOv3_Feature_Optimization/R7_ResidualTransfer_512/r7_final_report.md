# R7 Final Report: Stage 3 to Stage 4 Residual Feature Transfer

## Executive conclusion

R7 Best Epoch 6 achieved Macro-AUROC `0.802901` versus fresh R5 Best inference `0.803469`; R7−R5 delta `-0.000568` with Subject-cluster 95% CI `[-0.001113259792327881, -8.528530597686772e-05]`. The interval is below zero, indicating a small paired decrease in this Validation run; repeated Validation use and the unresolved R5 score replay discrepancy limit the strength of that inference.
Within the same R7 Best weights, disabling gamma yields Macro-AUROC `0.802953`; Fused−gamma=0 delta `-0.000052` with CI `[-0.00015056729316711425, 2.3426115512847892e-05]`. This directly measures the inference contribution of the branch in the trained R7 model.

## Required questions

1. R7 Best Macro-AUROC: **0.802901** at Epoch 6; Macro-AUPRC: **0.427874**.
2. Relative to the historically reported R5 Best Macro-AUROC `0.803514`, R7 is `-0.000613`. Paired comparisons use fresh same-loader R5 Best inference `0.803469`, giving a delta of `-0.000568` with CI `[-0.001113259792327881, -8.528530597686772e-05]` entirely below zero; this run provides no evidence of an AUROC gain and points to a small decrease.
3–5. Per-disease AUROC/AUPRC, sample sizes and paired intervals are in `r7_inference_ablation.csv` and `r7_vs_r5_per_class.csv`. Pneumonia/Consolidation/Lung Opacity are called out in those tables; the result should be read by class and with the CI.
6. Potential degradation in historically strong classes is also shown per disease in the same table; no class-wise significance claim is made without considering the ten-class multiple-comparison scope.
7. Best-epoch gamma is `-0.00020749128`; final-stop gamma is `-0.00023250337`. Its per-epoch trajectory is in `gamma_vs_epoch.png` and `r7_branch_diagnostics.csv`.
8. On all Validation images at Best, residual ratio `||gamma*T(F3)|| / ||F4||` mean/median/P90/max is `0.000654` / `0.000636` / `0.000867` / `0.001585`; cosine statistics are in the branch diagnostics CSV.
9. R7 Fused−R7 gamma=0 Macro-AUROC delta is `-0.000052`, 95% CI `[-0.00015056729316711425, 2.3426115512847892e-05]`; this is the inference-time direct branch effect on identical R7 weights.
10. R7−R5 Macro-AUROC paired 95% CI `[-0.001113259792327881, -8.528530597686772e-05]` is entirely negative for the fresh same-loader predictions; this suggests a small decrease in this run, subject to the R5 replay discrepancy, repeated Validation use and multiple-comparison limits. For Pneumonia, the point change is `-0.000781` with CI `[-0.00326806902885437, 0.002229174971580501]`.
11. R7 adds `294,913` trainable parameters. Best checkpoint size is `327,809,824` bytes; inference ZIP size is `104,652,777` bytes (limit passed). Standalone per-Study mean model forward latency is `39.32` ms; peak inference allocation is `0.244` GiB. Peak reserved training memory is R5 `1.57` / R7 `1.75` GiB; mean Train throughput through Best is R5 `22.14` / R7 `19.89` images/s, with Best-epoch cumulative updates `6037` / `6037`.
12. R7 Train BCE fell to `0.142555` while Validation BCE rose to `0.458949` at stop; Best was Epoch 6, consistent with late-epoch overfitting under the observed curves.
13. R7 did not improve over fresh R5 inference; its shared residual route produced a small negative Macro-AUROC difference, and disabling gamma on the same R7 weights changes Macro-AUROC by a near-zero amount with a CI crossing zero. This does not prove the route is generally harmful or that Stage 4 loses information.
14. The current result supports stopping the shared residual route. Any disease-selective transfer should be a separately pre-registered future experiment with an independent evaluation set. No attention or disease gate was added in R7.

## Limitations

R5's fresh Validation inference differs from the saved R5 probability table by max `|Δp|=0.009352`; its Macro-AUROC differs by `-0.000045` and max per-class AUROC by `0.000418`. Model weights, official Study keys and labels match, but the score-level difference's cause remains unresolved. Standalone per-Study R7 inference also differs slightly from batched Validation inference (`max |Δp|=0.006612`); this is consistent with batch-shape precision variation but does not prove the R5 cause. Paired comparisons use fresh R5 and R7 predictions generated together on the same Validation loader.
All model selection and inference comparisons use the previously reused Validation set. Subject-cluster bootstrap is paired and uses common draws, but repeated Validation use, multiple diseases, one training seed and Best-epoch selection limit inferential strength. `R7 gamma=0−R5` cannot isolate the causal training-time effect of the new branch. The analysis did not read Test.
