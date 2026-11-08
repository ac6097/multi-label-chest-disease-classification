# H3B training report (H3B_Weak_L2SP)

- Experiment: Weak L2-SP (0.1 x H3 lambda) on the identical Stage3/4 pretrained parameter set.
- Optimization: `{"mode": "l2sp", "lambda_sp": 2892779.942582381, "lambda_source": "0.1 x H3 calibrated lambda (2.89278e6)", "parameter_normalization": "mean squared deviation over regularized pretrained parameter elements (same as H3)"}`
- Status: 19/20 epochs; stop reason: `three_condition_early_stop`.
- Best epoch: 13; Validation Macro-AUROC: 0.791497; Validation BCE: 0.316011.
- Last epoch: 19; raw Train BCE: 0.251222; Validation BCE: 0.330165; Macro-AUROC: 0.780670.
- Raw BCE (unweighted Study mean) is used for every reported comparison; H2/H3 optimized objectives are logged separately.
- Test split was not read; model selection uses official Validation only.

| Epoch | Raw Train BCE | Valid BCE | Macro-AUROC | Macro-AUPRC | Updates |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.373440 | 0.347500 | 0.722715 | 0.330231 | 1005 |
| 2 | 0.335806 | 0.343508 | 0.740697 | 0.352291 | 1008 |
| 3 | 0.330722 | 0.335217 | 0.755311 | 0.360297 | 1006 |
| 4 | 0.325759 | 0.340883 | 0.760234 | 0.376953 | 1009 |
| 5 | 0.322775 | 0.328144 | 0.763791 | 0.373888 | 1004 |
| 6 | 0.318709 | 0.324306 | 0.774787 | 0.393581 | 1005 |
| 7 | 0.318066 | 0.326732 | 0.768119 | 0.384453 | 1007 |
| 8 | 0.314129 | 0.320936 | 0.780531 | 0.388785 | 1007 |
| 9 | 0.312390 | 0.328182 | 0.781403 | 0.392721 | 1005 |
| 10 | 0.309409 | 0.319661 | 0.783553 | 0.397300 | 1005 |
| 11 | 0.306187 | 0.321231 | 0.785289 | 0.407906 | 1006 |
| 12 | 0.303150 | 0.317842 | 0.779556 | 0.406379 | 1006 |
| 13 | 0.298430 | 0.316011 | 0.791497 | 0.413933 | 1006 |
| 14 | 0.292975 | 0.318791 | 0.787880 | 0.413190 | 1008 |
| 15 | 0.286452 | 0.320133 | 0.786985 | 0.417554 | 1006 |
| 16 | 0.278093 | 0.316115 | 0.786464 | 0.416105 | 1006 |
| 17 | 0.268941 | 0.322026 | 0.784327 | 0.405758 | 1007 |
| 18 | 0.259695 | 0.326178 | 0.783923 | 0.405459 | 1005 |
| 19 | 0.251222 | 0.330165 | 0.780670 | 0.403659 | 1005 |

