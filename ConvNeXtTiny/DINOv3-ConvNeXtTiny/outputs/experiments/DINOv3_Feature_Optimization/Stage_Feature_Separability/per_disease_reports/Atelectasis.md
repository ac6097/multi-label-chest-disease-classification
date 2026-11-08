# Atelectasis

- Validation positive / negative Studies: 193 / 666; Subjects: 228.

## Single-stage probe

| Backbone | Stage | AUROC (95% CI) | AUPRC (95% CI) | OOF AUROC | OOF AUPRC | C | Dim | F1@0.5 | Balanced Acc@0.5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R5 | Stage 1 | 0.6663 [0.6212, 0.7162] | 0.3202 [0.2736, 0.3762] | 0.7143 | 0.3634 | 1 | 96 | 0.0786 | 0.5030 |
| R5 | Stage 2 | 0.6773 [0.6267, 0.7318] | 0.3208 [0.2783, 0.3841] | 0.7295 | 0.3877 | 0.1 | 192 | 0.0873 | 0.5064 |
| R5 | Stage 3 | 0.7673 [0.7214, 0.8171] | 0.4693 [0.3987, 0.5512] | 0.8392 | 0.5631 | 0.01 | 384 | 0.3867 | 0.6135 |
| R5 | Stage 4 | 0.7724 [0.7311, 0.8219] | 0.4584 [0.3946, 0.5412] | 0.8997 | 0.7071 | 0.01 | 768 | 0.4136 | 0.6238 |
| R6 | Stage 1 | 0.6818 [0.6341, 0.7332] | 0.3365 [0.2855, 0.4120] | 0.7160 | 0.3634 | 1 | 96 | 0.0901 | 0.5116 |
| R6 | Stage 2 | 0.7022 [0.6536, 0.7519] | 0.3574 [0.3081, 0.4289] | 0.7355 | 0.3960 | 0.1 | 192 | 0.1481 | 0.5226 |
| R6 | Stage 3 | 0.7623 [0.7130, 0.8164] | 0.4722 [0.3976, 0.5533] | 0.8427 | 0.5665 | 0.01 | 384 | 0.3605 | 0.6013 |
| R6 | Stage 4 | 0.7784 [0.7367, 0.8270] | 0.4435 [0.3857, 0.5236] | 0.8949 | 0.6956 | 0.01 | 768 | 0.3704 | 0.6021 |

## Stage transitions and R6−R5

| Comparison | Δ AUROC (95% CI) | Δ AUPRC (95% CI) |
|---|---:|---:|
| R5 Stage2-Stage1 | 0.0110 [-0.0208, 0.0405] | 0.0006 [-0.0290, 0.0403] |
| R5 Stage3-Stage2 | 0.0900 [0.0518, 0.1324] | 0.1485 [0.0819, 0.2101] |
| R5 Stage4-Stage3 | 0.0051 [-0.0176, 0.0282] | -0.0109 [-0.0594, 0.0429] |
| R5 Stage4-Stage1 | 0.1061 [0.0614, 0.1530] | 0.1382 [0.0789, 0.2127] |
| R6 Stage2-Stage1 | 0.0204 [-0.0137, 0.0542] | 0.0209 [-0.0186, 0.0654] |
| R6 Stage3-Stage2 | 0.0601 [0.0231, 0.1011] | 0.1148 [0.0423, 0.1827] |
| R6 Stage4-Stage3 | 0.0161 [-0.0085, 0.0415] | -0.0287 [-0.0843, 0.0351] |
| R6 Stage4-Stage1 | 0.0966 [0.0543, 0.1392] | 0.1070 [0.0511, 0.1742] |
| R6−R5 Stage 1 | 0.0155 [-0.0058, 0.0379] | 0.0163 [-0.0118, 0.0566] |
| R6−R5 Stage 2 | 0.0249 [0.0027, 0.0460] | 0.0366 [0.0112, 0.0696] |
| R6−R5 Stage 3 | -0.0050 [-0.0330, 0.0220] | 0.0029 [-0.0428, 0.0500] |
| R6−R5 Stage 4 | 0.0060 [-0.0161, 0.0306] | -0.0149 [-0.0578, 0.0315] |

## Best Stage, joint probes, and PCA-64

| Backbone | Best Stage | Paired best−second AUROC (95% CI) | Advantage |
|---|---|---:|---|
| R5 | Stage 4 | 0.0051 [-0.0176, 0.0282] | No clear stage advantage |
| R6 | Stage 4 | 0.0161 [-0.0085, 0.0415] | No clear stage advantage |
| R5 Stage3+4−Stage3 | 0.0145 [-0.0065, 0.0362] | -0.0078 [-0.0539, 0.0400] |
| R5 Stage3+4−Stage4 | 0.0094 [0.0024, 0.0165] | 0.0030 [-0.0171, 0.0178] |
| R5 Stage2+3−Stage2 | 0.0841 [0.0443, 0.1259] | 0.1489 [0.0807, 0.2113] |
| R5 Stage2+3−Stage3 | -0.0059 [-0.0149, 0.0027] | 0.0004 [-0.0173, 0.0180] |
| R6 Stage3+4−Stage3 | 0.0212 [0.0004, 0.0432] | -0.0243 [-0.0737, 0.0341] |
| R6 Stage3+4−Stage4 | 0.0050 [-0.0030, 0.0128] | 0.0045 [-0.0123, 0.0192] |
| R6 Stage2+3−Stage2 | 0.0600 [0.0237, 0.1005] | 0.1122 [0.0380, 0.1834] |
| R6 Stage2+3−Stage3 | -0.0000 [-0.0057, 0.0058] | -0.0026 [-0.0121, 0.0104] |

## Disease-specific evidence and next direction

- R5: point-best Stage 4; Stage3+4−Stage4 AUROC +0.0094 [+0.0024, +0.0165]. Stage3+Stage4 provides positive paired evidence beyond Stage4 under this probe; prioritize a controlled Stage3→4 feature-transfer study.
- R6: point-best Stage 4; Stage3+4−Stage4 AUROC +0.0050 [-0.0030, +0.0128]. Stage3+Stage4 has a positive point change over Stage4, but its interval includes zero; no structural direction is confirmed.


PCA64 rows show the PCA-64 probe minus the corresponding raw Stage probe. PCA and scaling are fitted only on the current Train fold during CV, then on all Train for final Validation prediction.

| Backbone | Stage | Δ AUROC (95% CI) | Δ AUPRC (95% CI) |
|---|---|---:|---:|
| R5 | Stage 1 | 0.0042 [-0.0080, 0.0160] | -0.0035 [-0.0215, 0.0202] |
| R5 | Stage 2 | -0.0012 [-0.0213, 0.0183] | -0.0048 [-0.0324, 0.0159] |
| R5 | Stage 3 | 0.0034 [-0.0104, 0.0179] | 0.0172 [-0.0154, 0.0500] |
| R5 | Stage 4 | 0.0156 [0.0025, 0.0291] | 0.0102 [-0.0205, 0.0337] |
| R6 | Stage 1 | -0.0085 [-0.0209, 0.0026] | -0.0165 [-0.0319, 0.0067] |
| R6 | Stage 2 | -0.0094 [-0.0314, 0.0116] | -0.0197 [-0.0566, 0.0085] |
| R6 | Stage 3 | 0.0055 [-0.0112, 0.0221] | 0.0136 [-0.0209, 0.0460] |
| R6 | Stage 4 | 0.0084 [-0.0024, 0.0202] | 0.0075 [-0.0189, 0.0324] |

## Interpretation limits

These are Study-level linear probes on frozen features under deterministic ResizePad+GAP and the specified L2 Logistic Regression protocol. A higher Stage score is evidence of easier linear separability under this probe, not proof that a Stage contains uniquely important pathology information. Confidence intervals are exploratory because of the many disease/Stage/probe comparisons and repeated Validation use for historical checkpoint selection. No disease localization or causal claims are made.
