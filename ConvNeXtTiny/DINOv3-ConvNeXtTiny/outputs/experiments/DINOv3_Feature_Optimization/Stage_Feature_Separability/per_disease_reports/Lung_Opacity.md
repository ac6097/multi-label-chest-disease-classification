# Lung Opacity

- Validation positive / negative Studies: 214 / 645; Subjects: 228.

## Single-stage probe

| Backbone | Stage | AUROC (95% CI) | AUPRC (95% CI) | OOF AUROC | OOF AUPRC | C | Dim | F1@0.5 | Balanced Acc@0.5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R5 | Stage 1 | 0.6495 [0.5980, 0.6992] | 0.3764 [0.2771, 0.4629] | 0.6668 | 0.3662 | 1 | 96 | 0.0702 | 0.5140 |
| R5 | Stage 2 | 0.6479 [0.5901, 0.7059] | 0.3646 [0.2710, 0.4495] | 0.6800 | 0.3722 | 0.01 | 192 | 0.1143 | 0.5195 |
| R5 | Stage 3 | 0.7358 [0.6935, 0.7813] | 0.4551 [0.3920, 0.5398] | 0.7691 | 0.4784 | 0.01 | 384 | 0.3067 | 0.5765 |
| R5 | Stage 4 | 0.7210 [0.6750, 0.7651] | 0.4394 [0.3605, 0.5170] | 0.8559 | 0.6330 | 0.01 | 768 | 0.3664 | 0.5976 |
| R6 | Stage 1 | 0.6522 [0.5989, 0.7031] | 0.3605 [0.2741, 0.4440] | 0.6715 | 0.3703 | 10 | 96 | 0.0840 | 0.5125 |
| R6 | Stage 2 | 0.6801 [0.6277, 0.7351] | 0.4012 [0.3104, 0.4923] | 0.6946 | 0.3928 | 0.01 | 192 | 0.1575 | 0.5312 |
| R6 | Stage 3 | 0.7221 [0.6762, 0.7726] | 0.4527 [0.3848, 0.5424] | 0.7763 | 0.4914 | 0.01 | 384 | 0.3208 | 0.5850 |
| R6 | Stage 4 | 0.7268 [0.6848, 0.7718] | 0.4399 [0.3725, 0.5297] | 0.8513 | 0.6302 | 0.01 | 768 | 0.3923 | 0.6062 |

## Stage transitions and R6−R5

| Comparison | Δ AUROC (95% CI) | Δ AUPRC (95% CI) |
|---|---:|---:|
| R5 Stage2-Stage1 | -0.0016 [-0.0237, 0.0240] | -0.0118 [-0.0415, 0.0225] |
| R5 Stage3-Stage2 | 0.0879 [0.0516, 0.1294] | 0.0905 [0.0366, 0.1613] |
| R5 Stage4-Stage3 | -0.0148 [-0.0508, 0.0156] | -0.0156 [-0.0891, 0.0464] |
| R5 Stage4-Stage1 | 0.0715 [0.0198, 0.1241] | 0.0630 [-0.0119, 0.1451] |
| R6 Stage2-Stage1 | 0.0278 [0.0039, 0.0558] | 0.0407 [0.0005, 0.0823] |
| R6 Stage3-Stage2 | 0.0420 [0.0095, 0.0774] | 0.0515 [0.0033, 0.1147] |
| R6 Stage4-Stage3 | 0.0047 [-0.0238, 0.0342] | -0.0128 [-0.0570, 0.0295] |
| R6 Stage4-Stage1 | 0.0746 [0.0346, 0.1212] | 0.0795 [0.0193, 0.1576] |
| R6−R5 Stage 1 | 0.0028 [-0.0124, 0.0190] | -0.0159 [-0.0424, 0.0162] |
| R6−R5 Stage 2 | 0.0322 [0.0149, 0.0484] | 0.0366 [0.0139, 0.0670] |
| R6−R5 Stage 3 | -0.0137 [-0.0355, 0.0081] | -0.0024 [-0.0390, 0.0283] |
| R6−R5 Stage 4 | 0.0058 [-0.0223, 0.0377] | 0.0005 [-0.0472, 0.0635] |

## Best Stage, joint probes, and PCA-64

| Backbone | Best Stage | Paired best−second AUROC (95% CI) | Advantage |
|---|---|---:|---|
| R5 | Stage 3 | 0.0148 [-0.0156, 0.0508] | No clear stage advantage |
| R6 | Stage 4 | 0.0047 [-0.0238, 0.0342] | No clear stage advantage |
| R5 Stage3+4−Stage3 | -0.0074 [-0.0379, 0.0183] | 0.0104 [-0.0535, 0.0601] |
| R5 Stage3+4−Stage4 | 0.0074 [-0.0009, 0.0177] | 0.0260 [0.0047, 0.0509] |
| R5 Stage2+3−Stage2 | 0.0813 [0.0464, 0.1185] | 0.0786 [0.0316, 0.1446] |
| R5 Stage2+3−Stage3 | -0.0066 [-0.0153, 0.0018] | -0.0119 [-0.0355, 0.0115] |
| R6 Stage3+4−Stage3 | 0.0075 [-0.0179, 0.0323] | -0.0051 [-0.0433, 0.0352] |
| R6 Stage3+4−Stage4 | 0.0027 [-0.0063, 0.0114] | 0.0076 [-0.0126, 0.0297] |
| R6 Stage2+3−Stage2 | 0.0348 [0.0016, 0.0679] | 0.0310 [-0.0198, 0.0830] |
| R6 Stage2+3−Stage3 | -0.0072 [-0.0187, 0.0023] | -0.0205 [-0.0494, -0.0003] |

## Disease-specific evidence and next direction

- R5: point-best Stage 3; Stage3+4−Stage4 AUROC +0.0074 [-0.0009, +0.0177]. Stage3+Stage4 has a positive point change over Stage4, but its interval includes zero; no structural direction is confirmed.
- R6: point-best Stage 4; Stage3+4−Stage4 AUROC +0.0027 [-0.0063, +0.0114]. Stage3+Stage4 has a positive point change over Stage4, but its interval includes zero; no structural direction is confirmed.


PCA64 rows show the PCA-64 probe minus the corresponding raw Stage probe. PCA and scaling are fitted only on the current Train fold during CV, then on all Train for final Validation prediction.

| Backbone | Stage | Δ AUROC (95% CI) | Δ AUPRC (95% CI) |
|---|---|---:|---:|
| R5 | Stage 1 | 0.0022 [-0.0098, 0.0155] | 0.0078 [-0.0140, 0.0298] |
| R5 | Stage 2 | 0.0084 [-0.0060, 0.0220] | 0.0116 [-0.0120, 0.0392] |
| R5 | Stage 3 | -0.0130 [-0.0285, 0.0021] | 0.0054 [-0.0263, 0.0452] |
| R5 | Stage 4 | 0.0122 [-0.0030, 0.0270] | 0.0410 [0.0105, 0.0752] |
| R6 | Stage 1 | -0.0046 [-0.0216, 0.0143] | 0.0206 [-0.0121, 0.0553] |
| R6 | Stage 2 | 0.0017 [-0.0091, 0.0120] | 0.0076 [-0.0120, 0.0227] |
| R6 | Stage 3 | -0.0150 [-0.0322, 0.0031] | -0.0113 [-0.0393, 0.0169] |
| R6 | Stage 4 | -0.0059 [-0.0229, 0.0098] | 0.0015 [-0.0286, 0.0300] |

## Interpretation limits

These are Study-level linear probes on frozen features under deterministic ResizePad+GAP and the specified L2 Logistic Regression protocol. A higher Stage score is evidence of easier linear separability under this probe, not proof that a Stage contains uniquely important pathology information. Confidence intervals are exploratory because of the many disease/Stage/probe comparisons and repeated Validation use for historical checkpoint selection. No disease localization or causal claims are made.
