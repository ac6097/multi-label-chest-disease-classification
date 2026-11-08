# H2C training report (H2C_SingleWeight_1p50)

- Experiment: Single-image Study BCE weight 1.50 with normalized weighted Study mean (same implementation as H2).
- Optimization: `{"mode": "single_study_weight", "single_weight": 1.5}`
- Status: 10/20 epochs; stop reason: `three_condition_early_stop`.
- Best epoch: 7; Validation Macro-AUROC: 0.800501; Validation BCE: 0.331139.
- Last epoch: 10; raw Train BCE: 0.096643; Validation BCE: 0.558614; Macro-AUROC: 0.770104.
- Raw BCE (unweighted Study mean) is used for every reported comparison; H2/H3 optimized objectives are logged separately.
- Test split was not read; model selection uses official Validation only.

| Epoch | Raw Train BCE | Valid BCE | Macro-AUROC | Macro-AUPRC | Updates |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.372124 | 0.340552 | 0.734766 | 0.344520 | 1005 |
| 2 | 0.327935 | 0.336349 | 0.765966 | 0.380043 | 1008 |
| 3 | 0.314214 | 0.321428 | 0.783991 | 0.404310 | 1006 |
| 4 | 0.300318 | 0.342034 | 0.787058 | 0.406169 | 1009 |
| 5 | 0.285221 | 0.312721 | 0.789275 | 0.412766 | 1004 |
| 6 | 0.263051 | 0.315610 | 0.799444 | 0.419099 | 1005 |
| 7 | 0.231430 | 0.331139 | 0.800501 | 0.407754 | 1007 |
| 8 | 0.189353 | 0.395575 | 0.787012 | 0.397578 | 1007 |
| 9 | 0.142522 | 0.432720 | 0.764008 | 0.381899 | 1005 |
| 10 | 0.096643 | 0.558614 | 0.770104 | 0.390511 | 1005 |

