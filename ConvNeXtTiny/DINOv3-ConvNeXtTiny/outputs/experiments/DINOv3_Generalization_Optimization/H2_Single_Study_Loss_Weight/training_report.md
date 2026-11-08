# H2 training report (H2_Single_Study_Loss_Weight)

- Experiment: Single-image Study BCE weight 1.25 with normalized weighted Study mean.
- Optimization: `{"mode": "single_study_weight", "single_weight": 1.25}`
- Status: 9/20 epochs; stop reason: `three_condition_early_stop`.
- Best epoch: 6; Validation Macro-AUROC: 0.805038; Validation BCE: 0.311098.
- Last epoch: 9; raw Train BCE: 0.142542; Validation BCE: 0.448960; Macro-AUROC: 0.765304.
- Raw BCE (unweighted Study mean) is used for every reported comparison; H2/H3 optimized objectives are logged separately.
- Test split was not read; model selection uses official Validation only.

| Epoch | Raw Train BCE | Valid BCE | Macro-AUROC | Macro-AUPRC | Updates |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.371916 | 0.341440 | 0.734141 | 0.342928 | 1005 |
| 2 | 0.327557 | 0.334450 | 0.768067 | 0.376998 | 1008 |
| 3 | 0.313038 | 0.321185 | 0.786767 | 0.399242 | 1006 |
| 4 | 0.299610 | 0.344897 | 0.786403 | 0.407288 | 1009 |
| 5 | 0.284832 | 0.310464 | 0.796035 | 0.414063 | 1004 |
| 6 | 0.262883 | 0.311098 | 0.805038 | 0.424958 | 1005 |
| 7 | 0.230508 | 0.334842 | 0.798094 | 0.413028 | 1007 |
| 8 | 0.188532 | 0.397273 | 0.789175 | 0.400875 | 1007 |
| 9 | 0.142542 | 0.448960 | 0.765304 | 0.388584 | 1005 |

