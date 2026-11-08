# Cardiomegaly

- Validation positive / negative Studies: 177 / 682; Subjects: 228.

## Single-stage probe

| Backbone | Stage | AUROC (95% CI) | AUPRC (95% CI) | OOF AUROC | OOF AUPRC | C | Dim | F1@0.5 | Balanced Acc@0.5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R5 | Stage 1 | 0.6799 [0.6132, 0.7501] | 0.2881 [0.2216, 0.3755] | 0.7210 | 0.3613 | 0.1 | 96 | 0.0288 | 0.4879 |
| R5 | Stage 2 | 0.7010 [0.6345, 0.7681] | 0.3033 [0.2291, 0.4017] | 0.7296 | 0.3711 | 0.01 | 192 | 0.0751 | 0.5021 |
| R5 | Stage 3 | 0.7581 [0.7049, 0.8128] | 0.4135 [0.3093, 0.5177] | 0.8492 | 0.5989 | 0.01 | 384 | 0.3333 | 0.5911 |
| R5 | Stage 4 | 0.7655 [0.7197, 0.8136] | 0.4187 [0.3278, 0.4997] | 0.9066 | 0.7427 | 0.01 | 768 | 0.3672 | 0.6054 |
| R6 | Stage 1 | 0.6759 [0.6015, 0.7502] | 0.2868 [0.2109, 0.3875] | 0.7175 | 0.3517 | 1 | 96 | 0.0734 | 0.4984 |
| R6 | Stage 2 | 0.7038 [0.6371, 0.7681] | 0.3054 [0.2243, 0.4183] | 0.7299 | 0.3737 | 0.01 | 192 | 0.0922 | 0.5063 |
| R6 | Stage 3 | 0.7632 [0.7001, 0.8250] | 0.4113 [0.3115, 0.5121] | 0.8477 | 0.5912 | 0.01 | 384 | 0.3026 | 0.5770 |
| R6 | Stage 4 | 0.7646 [0.7113, 0.8200] | 0.4282 [0.3445, 0.5257] | 0.9021 | 0.7310 | 0.01 | 768 | 0.3194 | 0.5823 |

## Stage transitions and R6−R5

| Comparison | Δ AUROC (95% CI) | Δ AUPRC (95% CI) |
|---|---:|---:|
| R5 Stage2-Stage1 | 0.0211 [-0.0021, 0.0400] | 0.0152 [-0.0077, 0.0527] |
| R5 Stage3-Stage2 | 0.0571 [0.0226, 0.0933] | 0.1102 [0.0475, 0.1640] |
| R5 Stage4-Stage3 | 0.0074 [-0.0219, 0.0337] | 0.0052 [-0.0576, 0.0660] |
| R5 Stage4-Stage1 | 0.0856 [0.0305, 0.1325] | 0.1306 [0.0491, 0.1907] |
| R6 Stage2-Stage1 | 0.0279 [-0.0016, 0.0527] | 0.0186 [-0.0097, 0.0698] |
| R6 Stage3-Stage2 | 0.0594 [0.0230, 0.1070] | 0.1059 [0.0364, 0.1716] |
| R6 Stage4-Stage3 | 0.0013 [-0.0263, 0.0254] | 0.0169 [-0.0388, 0.0815] |
| R6 Stage4-Stage1 | 0.0886 [0.0290, 0.1439] | 0.1414 [0.0569, 0.2233] |
| R6−R5 Stage 1 | -0.0040 [-0.0204, 0.0134] | -0.0014 [-0.0228, 0.0306] |
| R6−R5 Stage 2 | 0.0028 [-0.0127, 0.0154] | 0.0021 [-0.0162, 0.0330] |
| R6−R5 Stage 3 | 0.0051 [-0.0150, 0.0293] | -0.0022 [-0.0418, 0.0463] |
| R6−R5 Stage 4 | -0.0009 [-0.0238, 0.0254] | 0.0095 [-0.0371, 0.0731] |

## Best Stage, joint probes, and PCA-64

| Backbone | Best Stage | Paired best−second AUROC (95% CI) | Advantage |
|---|---|---:|---|
| R5 | Stage 4 | 0.0074 [-0.0219, 0.0337] | No clear stage advantage |
| R6 | Stage 4 | 0.0013 [-0.0263, 0.0254] | No clear stage advantage |
| R5 Stage3+4−Stage3 | 0.0113 [-0.0132, 0.0340] | 0.0115 [-0.0464, 0.0730] |
| R5 Stage3+4−Stage4 | 0.0040 [-0.0052, 0.0145] | 0.0064 [-0.0062, 0.0246] |
| R5 Stage2+3−Stage2 | 0.0605 [0.0258, 0.0972] | 0.1115 [0.0454, 0.1679] |
| R5 Stage2+3−Stage3 | 0.0033 [-0.0043, 0.0107] | 0.0013 [-0.0127, 0.0169] |
| R6 Stage3+4−Stage3 | 0.0055 [-0.0202, 0.0278] | 0.0246 [-0.0287, 0.0834] |
| R6 Stage3+4−Stage4 | 0.0042 [-0.0040, 0.0114] | 0.0077 [-0.0087, 0.0207] |
| R6 Stage2+3−Stage2 | 0.0618 [0.0249, 0.1081] | 0.1044 [0.0360, 0.1691] |
| R6 Stage2+3−Stage3 | 0.0024 [-0.0032, 0.0076] | -0.0015 [-0.0112, 0.0094] |

## Disease-specific evidence and next direction

- R5: point-best Stage 4; Stage3+4−Stage4 AUROC +0.0040 [-0.0052, +0.0145]. Stage3+Stage4 has a positive point change over Stage4, but its interval includes zero; no structural direction is confirmed.
- R6: point-best Stage 4; Stage3+4−Stage4 AUROC +0.0042 [-0.0040, +0.0114]. Stage3+Stage4 has a positive point change over Stage4, but its interval includes zero; no structural direction is confirmed.


PCA64 rows show the PCA-64 probe minus the corresponding raw Stage probe. PCA and scaling are fitted only on the current Train fold during CV, then on all Train for final Validation prediction.

| Backbone | Stage | Δ AUROC (95% CI) | Δ AUPRC (95% CI) |
|---|---|---:|---:|
| R5 | Stage 1 | -0.0049 [-0.0103, 0.0001] | -0.0034 [-0.0091, 0.0070] |
| R5 | Stage 2 | -0.0160 [-0.0281, -0.0066] | -0.0103 [-0.0403, 0.0059] |
| R5 | Stage 3 | 0.0054 [-0.0102, 0.0235] | 0.0010 [-0.0358, 0.0405] |
| R5 | Stage 4 | 0.0072 [-0.0051, 0.0222] | 0.0121 [-0.0061, 0.0382] |
| R6 | Stage 1 | -0.0081 [-0.0222, 0.0076] | -0.0011 [-0.0172, 0.0139] |
| R6 | Stage 2 | -0.0155 [-0.0267, -0.0032] | -0.0160 [-0.0598, 0.0013] |
| R6 | Stage 3 | 0.0036 [-0.0103, 0.0179] | 0.0143 [-0.0246, 0.0528] |
| R6 | Stage 4 | 0.0062 [-0.0063, 0.0189] | 0.0173 [-0.0079, 0.0377] |

## Interpretation limits

These are Study-level linear probes on frozen features under deterministic ResizePad+GAP and the specified L2 Logistic Regression protocol. A higher Stage score is evidence of easier linear separability under this probe, not proof that a Stage contains uniquely important pathology information. Confidence intervals are exploratory because of the many disease/Stage/probe comparisons and repeated Validation use for historical checkpoint selection. No disease localization or causal claims are made.
