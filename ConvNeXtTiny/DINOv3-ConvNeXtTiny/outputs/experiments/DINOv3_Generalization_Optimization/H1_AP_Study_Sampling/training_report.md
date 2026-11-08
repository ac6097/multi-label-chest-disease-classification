# H1 training report (H1_AP_Study_Sampling)

- Experiment: Study-level weighted sampling toward AP-only Studies (target 55%).
- Optimization: `{"mode": "ap_sampling", "target_ap_fraction": 0.55, "ap_weight": 1.2}`
- Status: 7/20 epochs; stop reason: `three_condition_early_stop`.
- Best epoch: 4; Validation Macro-AUROC: 0.785835; Validation BCE: 0.329965.
- Last epoch: 7; raw Train BCE: 0.167028; Validation BCE: 0.438363; Macro-AUROC: 0.755185.
- Raw BCE (unweighted Study mean) is used for every reported comparison; H2/H3 optimized objectives are logged separately.
- Test split was not read; model selection uses official Validation only.

| Epoch | Raw Train BCE | Valid BCE | Macro-AUROC | Macro-AUPRC | Updates |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.377232 | 0.345739 | 0.741519 | 0.340963 | 1005 |
| 2 | 0.327954 | 0.333168 | 0.764572 | 0.381208 | 1008 |
| 3 | 0.307744 | 0.321228 | 0.780867 | 0.391348 | 1006 |
| 4 | 0.285889 | 0.329965 | 0.785835 | 0.388620 | 1009 |
| 5 | 0.257212 | 0.368116 | 0.766364 | 0.379401 | 1004 |
| 6 | 0.213046 | 0.387209 | 0.770359 | 0.387708 | 1005 |
| 7 | 0.167028 | 0.438363 | 0.755185 | 0.365224 | 1007 |

