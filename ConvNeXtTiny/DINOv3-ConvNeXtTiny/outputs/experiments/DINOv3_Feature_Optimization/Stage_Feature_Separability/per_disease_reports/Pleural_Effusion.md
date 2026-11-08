# Pleural Effusion

- Validation positive / negative Studies: 260 / 599; Subjects: 228.

## Single-stage probe

| Backbone | Stage | AUROC (95% CI) | AUPRC (95% CI) | OOF AUROC | OOF AUPRC | C | Dim | F1@0.5 | Balanced Acc@0.5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R5 | Stage 1 | 0.7902 [0.7188, 0.8534] | 0.5546 [0.4420, 0.6772] | 0.7940 | 0.5199 | 10 | 96 | 0.4289 | 0.6195 |
| R5 | Stage 2 | 0.8351 [0.7690, 0.8915] | 0.6227 [0.5071, 0.7431] | 0.8263 | 0.5777 | 10 | 192 | 0.5854 | 0.7046 |
| R5 | Stage 3 | 0.9000 [0.8492, 0.9389] | 0.7709 [0.6792, 0.8468] | 0.9363 | 0.8189 | 0.01 | 384 | 0.7143 | 0.7948 |
| R5 | Stage 4 | 0.9059 [0.8605, 0.9426] | 0.7730 [0.6893, 0.8483] | 0.9651 | 0.8989 | 0.01 | 768 | 0.7429 | 0.8166 |
| R6 | Stage 1 | 0.7983 [0.7346, 0.8553] | 0.5816 [0.4808, 0.6920] | 0.7938 | 0.5210 | 10 | 96 | 0.4905 | 0.6505 |
| R6 | Stage 2 | 0.8306 [0.7637, 0.8888] | 0.6517 [0.5362, 0.7541] | 0.8381 | 0.5987 | 1 | 192 | 0.5887 | 0.7064 |
| R6 | Stage 3 | 0.8980 [0.8419, 0.9397] | 0.7720 [0.6675, 0.8521] | 0.9388 | 0.8257 | 0.01 | 384 | 0.7234 | 0.8012 |
| R6 | Stage 4 | 0.9056 [0.8527, 0.9451] | 0.8015 [0.7161, 0.8693] | 0.9609 | 0.8885 | 0.001 | 768 | 0.7305 | 0.8054 |

## Stage transitions and R6−R5

| Comparison | Δ AUROC (95% CI) | Δ AUPRC (95% CI) |
|---|---:|---:|
| R5 Stage2-Stage1 | 0.0448 [0.0188, 0.0706] | 0.0682 [0.0209, 0.1267] |
| R5 Stage3-Stage2 | 0.0650 [0.0319, 0.0950] | 0.1482 [0.0696, 0.2060] |
| R5 Stage4-Stage3 | 0.0058 [-0.0050, 0.0166] | 0.0020 [-0.0339, 0.0316] |
| R5 Stage4-Stage1 | 0.1156 [0.0747, 0.1543] | 0.2184 [0.1220, 0.3008] |
| R6 Stage2-Stage1 | 0.0323 [0.0019, 0.0633] | 0.0702 [-0.0007, 0.1351] |
| R6 Stage3-Stage2 | 0.0675 [0.0414, 0.0947] | 0.1203 [0.0702, 0.1765] |
| R6 Stage4-Stage3 | 0.0076 [-0.0038, 0.0177] | 0.0295 [-0.0020, 0.0572] |
| R6 Stage4-Stage1 | 0.1074 [0.0724, 0.1413] | 0.2199 [0.1403, 0.2890] |
| R6−R5 Stage 1 | 0.0080 [-0.0092, 0.0236] | 0.0270 [-0.0189, 0.0579] |
| R6−R5 Stage 2 | -0.0045 [-0.0260, 0.0148] | 0.0290 [-0.0305, 0.0660] |
| R6−R5 Stage 3 | -0.0020 [-0.0149, 0.0088] | 0.0011 [-0.0226, 0.0218] |
| R6−R5 Stage 4 | -0.0002 [-0.0113, 0.0091] | 0.0286 [0.0007, 0.0575] |

## Best Stage, joint probes, and PCA-64

| Backbone | Best Stage | Paired best−second AUROC (95% CI) | Advantage |
|---|---|---:|---|
| R5 | Stage 4 | 0.0058 [-0.0050, 0.0166] | No clear stage advantage |
| R6 | Stage 4 | 0.0076 [-0.0038, 0.0177] | No clear stage advantage |
| R5 Stage3+4−Stage3 | 0.0079 [-0.0025, 0.0174] | 0.0110 [-0.0201, 0.0400] |
| R5 Stage3+4−Stage4 | 0.0021 [-0.0013, 0.0054] | 0.0090 [-0.0024, 0.0216] |
| R5 Stage2+3−Stage2 | 0.0656 [0.0339, 0.0953] | 0.1469 [0.0709, 0.2055] |
| R5 Stage2+3−Stage3 | 0.0006 [-0.0025, 0.0043] | -0.0013 [-0.0089, 0.0088] |
| R6 Stage3+4−Stage3 | 0.0086 [-0.0022, 0.0183] | 0.0309 [-0.0008, 0.0593] |
| R6 Stage3+4−Stage4 | 0.0010 [-0.0005, 0.0027] | 0.0014 [-0.0044, 0.0073] |
| R6 Stage2+3−Stage2 | 0.0682 [0.0420, 0.0954] | 0.1219 [0.0708, 0.1769] |
| R6 Stage2+3−Stage3 | 0.0008 [-0.0015, 0.0031] | 0.0016 [-0.0058, 0.0076] |

## Disease-specific evidence and next direction

- R5: point-best Stage 4; Stage3+4−Stage4 AUROC +0.0021 [-0.0013, +0.0054]. Stage3+Stage4 has a positive point change over Stage4, but its interval includes zero; no structural direction is confirmed.
- R6: point-best Stage 4; Stage3+4−Stage4 AUROC +0.0010 [-0.0005, +0.0027]. Stage3+Stage4 has a positive point change over Stage4, but its interval includes zero; no structural direction is confirmed.


PCA64 rows show the PCA-64 probe minus the corresponding raw Stage probe. PCA and scaling are fitted only on the current Train fold during CV, then on all Train for final Validation prediction.

| Backbone | Stage | Δ AUROC (95% CI) | Δ AUPRC (95% CI) |
|---|---|---:|---:|
| R5 | Stage 1 | -0.0036 [-0.0177, 0.0098] | -0.0092 [-0.0324, 0.0234] |
| R5 | Stage 2 | -0.0140 [-0.0294, 0.0014] | -0.0124 [-0.0483, 0.0194] |
| R5 | Stage 3 | -0.0095 [-0.0165, -0.0028] | -0.0252 [-0.0512, -0.0011] |
| R5 | Stage 4 | 0.0046 [0.0001, 0.0091] | 0.0204 [0.0049, 0.0359] |
| R6 | Stage 1 | -0.0119 [-0.0251, 0.0038] | -0.0354 [-0.0696, 0.0262] |
| R6 | Stage 2 | -0.0125 [-0.0300, 0.0071] | -0.0492 [-0.0817, -0.0057] |
| R6 | Stage 3 | -0.0066 [-0.0123, -0.0003] | -0.0215 [-0.0485, 0.0117] |
| R6 | Stage 4 | 0.0014 [-0.0005, 0.0035] | 0.0028 [-0.0033, 0.0091] |

## Interpretation limits

These are Study-level linear probes on frozen features under deterministic ResizePad+GAP and the specified L2 Logistic Regression protocol. A higher Stage score is evidence of easier linear separability under this probe, not proof that a Stage contains uniquely important pathology information. Confidence intervals are exploratory because of the many disease/Stage/probe comparisons and repeated Validation use for historical checkpoint selection. No disease localization or causal claims are made.
