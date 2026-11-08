# Pneumothorax

- Validation positive / negative Studies: 37 / 822; Subjects: 228.

## Single-stage probe

| Backbone | Stage | AUROC (95% CI) | AUPRC (95% CI) | OOF AUROC | OOF AUPRC | C | Dim | F1@0.5 | Balanced Acc@0.5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R5 | Stage 1 | 0.7473 [0.6348, 0.8542] | 0.1043 [0.0480, 0.1925] | 0.7629 | 0.1776 | 10 | 96 | 0.0000 | 0.5000 |
| R5 | Stage 2 | 0.7507 [0.5871, 0.9050] | 0.1621 [0.0574, 0.3656] | 0.7741 | 0.1871 | 0.1 | 192 | 0.0000 | 0.4994 |
| R5 | Stage 3 | 0.8248 [0.7286, 0.9058] | 0.1868 [0.0917, 0.3366] | 0.8849 | 0.4433 | 0.01 | 384 | 0.0476 | 0.5111 |
| R5 | Stage 4 | 0.8048 [0.6978, 0.8868] | 0.2261 [0.0941, 0.4468] | 0.9435 | 0.6738 | 0.01 | 768 | 0.1633 | 0.5492 |
| R6 | Stage 1 | 0.7706 [0.6516, 0.8894] | 0.1272 [0.0558, 0.2474] | 0.7601 | 0.1700 | 0.1 | 96 | 0.0000 | 0.5000 |
| R6 | Stage 2 | 0.7667 [0.6152, 0.9048] | 0.2181 [0.0585, 0.4825] | 0.7801 | 0.1879 | 1 | 192 | 0.0500 | 0.5123 |
| R6 | Stage 3 | 0.8181 [0.7106, 0.9108] | 0.2527 [0.1243, 0.4001] | 0.8906 | 0.4550 | 0.01 | 384 | 0.2041 | 0.5633 |
| R6 | Stage 4 | 0.8224 [0.7290, 0.8927] | 0.3634 [0.2108, 0.4940] | 0.9428 | 0.6574 | 0.01 | 768 | 0.3158 | 0.6149 |

## Stage transitions and R6−R5

| Comparison | Δ AUROC (95% CI) | Δ AUPRC (95% CI) |
|---|---:|---:|
| R5 Stage2-Stage1 | 0.0034 [-0.0746, 0.0906] | 0.0579 [-0.0131, 0.2059] |
| R5 Stage3-Stage2 | 0.0741 [-0.0411, 0.1765] | 0.0247 [-0.1136, 0.1337] |
| R5 Stage4-Stage3 | -0.0200 [-0.0975, 0.0351] | 0.0393 [-0.0592, 0.1729] |
| R5 Stage4-Stage1 | 0.0575 [-0.0426, 0.1290] | 0.1218 [0.0140, 0.2940] |
| R6 Stage2-Stage1 | -0.0038 [-0.0656, 0.0575] | 0.0909 [-0.0168, 0.2627] |
| R6 Stage3-Stage2 | 0.0514 [-0.0198, 0.1223] | 0.0345 [-0.1146, 0.1518] |
| R6 Stage4-Stage3 | 0.0042 [-0.0498, 0.0496] | 0.1107 [0.0184, 0.1820] |
| R6 Stage4-Stage1 | 0.0518 [-0.0489, 0.1272] | 0.2362 [0.1244, 0.3167] |
| R6−R5 Stage 1 | 0.0233 [-0.0061, 0.0650] | 0.0229 [-0.0016, 0.0693] |
| R6−R5 Stage 2 | 0.0160 [-0.0404, 0.0560] | 0.0560 [-0.0124, 0.1633] |
| R6−R5 Stage 3 | -0.0066 [-0.0534, 0.0250] | 0.0659 [-0.0071, 0.1431] |
| R6−R5 Stage 4 | 0.0176 [-0.0310, 0.0640] | 0.1373 [-0.0100, 0.2289] |

## Best Stage, joint probes, and PCA-64

| Backbone | Best Stage | Paired best−second AUROC (95% CI) | Advantage |
|---|---|---:|---|
| R5 | Stage 3 | 0.0200 [-0.0351, 0.0975] | No clear stage advantage |
| R6 | Stage 4 | 0.0042 [-0.0498, 0.0496] | No clear stage advantage |
| R5 Stage3+4−Stage3 | -0.0339 [-0.1260, 0.0294] | 0.0381 [-0.0725, 0.1714] |
| R5 Stage3+4−Stage4 | -0.0139 [-0.0406, 0.0046] | -0.0012 [-0.0879, 0.0398] |
| R5 Stage2+3−Stage2 | 0.0608 [-0.0487, 0.1570] | 0.0150 [-0.1109, 0.1085] |
| R5 Stage2+3−Stage3 | -0.0133 [-0.0265, 0.0010] | -0.0096 [-0.0331, 0.0133] |
| R6 Stage3+4−Stage3 | -0.0123 [-0.0729, 0.0329] | 0.0893 [-0.0065, 0.1423] |
| R6 Stage3+4−Stage4 | -0.0166 [-0.0365, -0.0006] | -0.0214 [-0.0538, -0.0077] |
| R6 Stage2+3−Stage2 | 0.0464 [-0.0244, 0.1124] | 0.0028 [-0.1467, 0.1198] |
| R6 Stage2+3−Stage3 | -0.0050 [-0.0164, 0.0081] | -0.0318 [-0.0592, -0.0046] |

## Disease-specific evidence and next direction

- R5: point-best Stage 3; Stage3+4−Stage4 AUROC -0.0139 [-0.0406, +0.0046]. Current stage and joint-probe intervals do not isolate a clear bottleneck; retain this disease as an open question for a pre-registered follow-up.
- R6: point-best Stage 4; Stage3+4−Stage4 AUROC -0.0166 [-0.0365, -0.0006]. Current stage and joint-probe intervals do not isolate a clear bottleneck; retain this disease as an open question for a pre-registered follow-up.


PCA64 rows show the PCA-64 probe minus the corresponding raw Stage probe. PCA and scaling are fitted only on the current Train fold during CV, then on all Train for final Validation prediction.

| Backbone | Stage | Δ AUROC (95% CI) | Δ AUPRC (95% CI) |
|---|---|---:|---:|
| R5 | Stage 1 | -0.0305 [-0.0830, 0.0348] | 0.0072 [-0.0172, 0.0642] |
| R5 | Stage 2 | -0.0269 [-0.0744, 0.0143] | -0.0585 [-0.1728, 0.0177] |
| R5 | Stage 3 | 0.0178 [-0.0218, 0.0504] | 0.0291 [-0.0219, 0.1120] |
| R5 | Stage 4 | 0.0388 [0.0161, 0.0676] | 0.0991 [0.0248, 0.1501] |
| R6 | Stage 1 | -0.0090 [-0.0199, 0.0047] | -0.0012 [-0.0083, 0.0119] |
| R6 | Stage 2 | -0.0586 [-0.1360, 0.0221] | -0.0937 [-0.2751, 0.0181] |
| R6 | Stage 3 | 0.0131 [-0.0310, 0.0554] | 0.0158 [-0.0713, 0.0863] |
| R6 | Stage 4 | 0.0184 [-0.0028, 0.0476] | 0.0087 [-0.0508, 0.0705] |

## Interpretation limits

These are Study-level linear probes on frozen features under deterministic ResizePad+GAP and the specified L2 Logistic Regression protocol. A higher Stage score is evidence of easier linear separability under this probe, not proof that a Stage contains uniquely important pathology information. Confidence intervals are exploratory because of the many disease/Stage/probe comparisons and repeated Validation use for historical checkpoint selection. No disease localization or causal claims are made.
