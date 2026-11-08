# H2B training report (H2B_SingleWeight_1p10)

- Experiment: Single-image Study BCE weight 1.10 with normalized weighted Study mean (same implementation as H2).
- Optimization: `{"mode": "single_study_weight", "single_weight": 1.1}`
- Status: 9/20 epochs; stop reason: `three_condition_early_stop`.
- Best epoch: 6; Validation Macro-AUROC: 0.803685; Validation BCE: 0.311292.
- Last epoch: 9; raw Train BCE: 0.142132; Validation BCE: 0.448062; Macro-AUROC: 0.767252.
- Raw BCE (unweighted Study mean) is used for every reported comparison; H2/H3 optimized objectives are logged separately.
- Test split was not read; model selection uses official Validation only.

| Epoch | Raw Train BCE | Valid BCE | Macro-AUROC | Macro-AUPRC | Updates |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.371849 | 0.342483 | 0.733622 | 0.342313 | 1005 |
| 2 | 0.327411 | 0.331638 | 0.769212 | 0.375838 | 1008 |
| 3 | 0.312457 | 0.321110 | 0.786431 | 0.399228 | 1006 |
| 4 | 0.299399 | 0.345484 | 0.786284 | 0.408435 | 1009 |
| 5 | 0.284589 | 0.310009 | 0.797052 | 0.414085 | 1004 |
| 6 | 0.262591 | 0.311292 | 0.803685 | 0.427324 | 1005 |
| 7 | 0.230477 | 0.337940 | 0.796290 | 0.414158 | 1007 |
| 8 | 0.188488 | 0.392317 | 0.790945 | 0.409949 | 1007 |
| 9 | 0.142132 | 0.448062 | 0.767252 | 0.389223 | 1005 |

