# Enlarged Cardiomediastinum

- Validation positive / negative Studies: 24 / 835; Subjects: 228.

## Single-stage probe

| Backbone | Stage | AUROC (95% CI) | AUPRC (95% CI) | OOF AUROC | OOF AUPRC | C | Dim | F1@0.5 | Balanced Acc@0.5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R5 | Stage 1 | 0.7178 [0.6023, 0.8110] | 0.0716 [0.0296, 0.2315] | 0.6699 | 0.0580 | 0.01 | 96 | 0.0000 | 0.5000 |
| R5 | Stage 2 | 0.6730 [0.5257, 0.7881] | 0.0627 [0.0297, 0.1265] | 0.6726 | 0.0603 | 0.01 | 192 | 0.0000 | 0.5000 |
| R5 | Stage 3 | 0.7110 [0.5755, 0.8081] | 0.0802 [0.0269, 0.1835] | 0.7399 | 0.0901 | 0.01 | 384 | 0.0000 | 0.5000 |
| R5 | Stage 4 | 0.7313 [0.5900, 0.8358] | 0.1137 [0.0295, 0.3141] | 0.8379 | 0.2308 | 0.01 | 768 | 0.0000 | 0.4994 |
| R6 | Stage 1 | 0.6951 [0.5765, 0.7930] | 0.0672 [0.0275, 0.2322] | 0.6708 | 0.0585 | 0.01 | 96 | 0.0000 | 0.5000 |
| R6 | Stage 2 | 0.6799 [0.5460, 0.7946] | 0.0658 [0.0271, 0.1551] | 0.6707 | 0.0601 | 0.01 | 192 | 0.0000 | 0.5000 |
| R6 | Stage 3 | 0.7184 [0.5505, 0.8360] | 0.0761 [0.0273, 0.1783] | 0.7409 | 0.0924 | 0.01 | 384 | 0.0000 | 0.5000 |
| R6 | Stage 4 | 0.7096 [0.5328, 0.8395] | 0.0745 [0.0234, 0.1976] | 0.8168 | 0.2081 | 0.001 | 768 | 0.0000 | 0.4994 |

## Stage transitions and R6−R5

| Comparison | Δ AUROC (95% CI) | Δ AUPRC (95% CI) |
|---|---:|---:|
| R5 Stage2-Stage1 | -0.0448 [-0.1174, 0.0134] | -0.0089 [-0.1354, 0.0329] |
| R5 Stage3-Stage2 | 0.0380 [-0.0801, 0.1256] | 0.0175 [-0.0509, 0.0864] |
| R5 Stage4-Stage3 | 0.0203 [-0.0764, 0.1302] | 0.0335 [-0.0356, 0.1557] |
| R5 Stage4-Stage1 | 0.0135 [-0.1431, 0.1455] | 0.0421 [-0.1367, 0.2136] |
| R6 Stage2-Stage1 | -0.0152 [-0.0885, 0.0451] | -0.0014 [-0.1345, 0.0529] |
| R6 Stage3-Stage2 | 0.0385 [-0.0701, 0.1234] | 0.0104 [-0.0264, 0.0603] |
| R6 Stage4-Stage3 | -0.0088 [-0.0871, 0.0661] | -0.0016 [-0.0475, 0.0453] |
| R6 Stage4-Stage1 | 0.0145 [-0.1672, 0.1455] | 0.0073 [-0.1453, 0.1030] |
| R6−R5 Stage 1 | -0.0227 [-0.0513, 0.0010] | -0.0044 [-0.0363, 0.0470] |
| R6−R5 Stage 2 | 0.0069 [-0.0253, 0.0479] | 0.0031 [-0.0280, 0.0521] |
| R6−R5 Stage 3 | 0.0074 [-0.1027, 0.1045] | -0.0041 [-0.0287, 0.0261] |
| R6−R5 Stage 4 | -0.0217 [-0.1094, 0.0458] | -0.0392 [-0.1349, 0.0009] |

## Best Stage, joint probes, and PCA-64

| Backbone | Best Stage | Paired best−second AUROC (95% CI) | Advantage |
|---|---|---:|---|
| R5 | Stage 4 | 0.0135 [-0.1431, 0.1455] | No clear stage advantage |
| R6 | Stage 3 | 0.0088 [-0.0661, 0.0871] | No clear stage advantage |
| R5 Stage3+4−Stage3 | -0.0156 [-0.1076, 0.0744] | 0.0100 [-0.0303, 0.0814] |
| R5 Stage3+4−Stage4 | -0.0359 [-0.0762, -0.0081] | -0.0236 [-0.1030, 0.0286] |
| R5 Stage2+3−Stage2 | 0.0415 [-0.0575, 0.1263] | 0.0117 [-0.0475, 0.0636] |
| R5 Stage2+3−Stage3 | 0.0035 [-0.0258, 0.0448] | -0.0059 [-0.0384, 0.0163] |
| R6 Stage3+4−Stage3 | -0.0005 [-0.0759, 0.0679] | 0.0026 [-0.0391, 0.0507] |
| R6 Stage3+4−Stage4 | 0.0083 [-0.0048, 0.0234] | 0.0042 [-0.0033, 0.0145] |
| R6 Stage2+3−Stage2 | 0.0320 [-0.0741, 0.1056] | 0.0135 [-0.0246, 0.0587] |
| R6 Stage2+3−Stage3 | -0.0065 [-0.0345, 0.0148] | 0.0031 [-0.0150, 0.0213] |

## Disease-specific evidence and next direction

- R5: point-best Stage 4; Stage3+4−Stage4 AUROC -0.0359 [-0.0762, -0.0081]. Current stage and joint-probe intervals do not isolate a clear bottleneck; retain this disease as an open question for a pre-registered follow-up.
- R6: point-best Stage 3; Stage3+4−Stage4 AUROC +0.0083 [-0.0048, +0.0234]. Stage3+Stage4 has a positive point change over Stage4, but its interval includes zero; no structural direction is confirmed.


PCA64 rows show the PCA-64 probe minus the corresponding raw Stage probe. PCA and scaling are fitted only on the current Train fold during CV, then on all Train for final Validation prediction.

| Backbone | Stage | Δ AUROC (95% CI) | Δ AUPRC (95% CI) |
|---|---|---:|---:|
| R5 | Stage 1 | -0.0017 [-0.0045, 0.0014] | -0.0011 [-0.0209, 0.0001] |
| R5 | Stage 2 | -0.0101 [-0.0321, 0.0106] | 0.0015 [-0.0226, 0.0516] |
| R5 | Stage 3 | 0.0063 [-0.0553, 0.0543] | 0.0045 [-0.0323, 0.0432] |
| R5 | Stage 4 | -0.0874 [-0.1389, -0.0500] | -0.0345 [-0.1260, 0.0086] |
| R6 | Stage 1 | 0.0000 [-0.0023, 0.0020] | 0.0001 [-0.0003, 0.0007] |
| R6 | Stage 2 | -0.0007 [-0.0210, 0.0216] | -0.0013 [-0.0250, 0.0189] |
| R6 | Stage 3 | 0.0161 [-0.0409, 0.0744] | 0.0373 [-0.0074, 0.1709] |
| R6 | Stage 4 | 0.0062 [-0.0079, 0.0238] | 0.0117 [-0.0051, 0.0821] |

## Interpretation limits

These are Study-level linear probes on frozen features under deterministic ResizePad+GAP and the specified L2 Logistic Regression protocol. A higher Stage score is evidence of easier linear separability under this probe, not proof that a Stage contains uniquely important pathology information. Confidence intervals are exploratory because of the many disease/Stage/probe comparisons and repeated Validation use for historical checkpoint selection. No disease localization or causal claims are made.
