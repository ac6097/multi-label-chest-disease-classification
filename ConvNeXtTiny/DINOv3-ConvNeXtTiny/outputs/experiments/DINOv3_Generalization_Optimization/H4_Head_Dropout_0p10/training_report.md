# H4 training report (H4_Head_Dropout_0p10)

- Experiment: Dropout p=0.10 on the final pooled feature immediately before the Global-only linear classifier; no L2-SP, no sampler or other head changes.
- Optimization: `{"mode": "head_dropout", "head_dropout": 0.1}`
- Status: 9/20 epochs; stop reason: `three_condition_early_stop`.
- Best epoch: 6; Validation Macro-AUROC: 0.803069; Validation BCE: 0.313042.
- Last epoch: 9; raw Train BCE: 0.160208; Validation BCE: 0.426254; Macro-AUROC: 0.778455.
- Raw BCE (unweighted Study mean) is used for every reported comparison; H2/H3 optimized objectives are logged separately.
- Test split was not read; model selection uses official Validation only.

| Epoch | Raw Train BCE | Valid BCE | Macro-AUROC | Macro-AUPRC | Updates |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.374598 | 0.346957 | 0.727805 | 0.336483 | 1005 |
| 2 | 0.326994 | 0.327464 | 0.763311 | 0.371884 | 1008 |
| 3 | 0.312812 | 0.323573 | 0.786946 | 0.399094 | 1006 |
| 4 | 0.300763 | 0.338755 | 0.786418 | 0.414192 | 1009 |
| 5 | 0.286697 | 0.310011 | 0.796823 | 0.418427 | 1004 |
| 6 | 0.267159 | 0.313042 | 0.803069 | 0.425843 | 1005 |
| 7 | 0.238858 | 0.331610 | 0.799693 | 0.417351 | 1007 |
| 8 | 0.202648 | 0.373797 | 0.790386 | 0.402538 | 1007 |
| 9 | 0.160208 | 0.426254 | 0.778455 | 0.394617 | 1005 |

