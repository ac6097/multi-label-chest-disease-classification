# Consolidation

- Validation positive / negative Studies: 46 / 813; Subjects: 228.

## Single-stage probe

| Backbone | Stage | AUROC (95% CI) | AUPRC (95% CI) | OOF AUROC | OOF AUPRC | C | Dim | F1@0.5 | Balanced Acc@0.5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R5 | Stage 1 | 0.6579 [0.5698, 0.7332] | 0.0824 [0.0534, 0.1271] | 0.7054 | 0.1020 | 0.1 | 96 | 0.0000 | 0.5000 |
| R5 | Stage 2 | 0.6890 [0.6053, 0.7595] | 0.0876 [0.0501, 0.1447] | 0.7238 | 0.1097 | 0.01 | 192 | 0.0000 | 0.5000 |
| R5 | Stage 3 | 0.7220 [0.6327, 0.7929] | 0.1442 [0.0682, 0.2130] | 0.8075 | 0.1876 | 0.01 | 384 | 0.0400 | 0.5090 |
| R5 | Stage 4 | 0.7806 [0.7059, 0.8367] | 0.1591 [0.0999, 0.2388] | 0.8866 | 0.3496 | 0.01 | 768 | 0.1587 | 0.5470 |
| R6 | Stage 1 | 0.6697 [0.5897, 0.7399] | 0.0842 [0.0510, 0.1400] | 0.7075 | 0.1053 | 0.1 | 96 | 0.0000 | 0.5000 |
| R6 | Stage 2 | 0.6857 [0.5956, 0.7621] | 0.0977 [0.0479, 0.1789] | 0.7322 | 0.1165 | 0.01 | 192 | 0.0000 | 0.5000 |
| R6 | Stage 3 | 0.7233 [0.6360, 0.7968] | 0.1492 [0.0623, 0.2549] | 0.8094 | 0.1819 | 0.01 | 384 | 0.0408 | 0.5096 |
| R6 | Stage 4 | 0.7531 [0.6694, 0.8132] | 0.1747 [0.0781, 0.2877] | 0.8870 | 0.3415 | 0.01 | 768 | 0.1379 | 0.5386 |

## Stage transitions and R6−R5

| Comparison | Δ AUROC (95% CI) | Δ AUPRC (95% CI) |
|---|---:|---:|
| R5 Stage2-Stage1 | 0.0312 [-0.0210, 0.0725] | 0.0051 [-0.0144, 0.0314] |
| R5 Stage3-Stage2 | 0.0330 [-0.0228, 0.0853] | 0.0566 [0.0027, 0.1049] |
| R5 Stage4-Stage3 | 0.0586 [0.0044, 0.1224] | 0.0149 [-0.0445, 0.1121] |
| R5 Stage4-Stage1 | 0.1228 [0.0513, 0.1931] | 0.0766 [0.0300, 0.1520] |
| R6 Stage2-Stage1 | 0.0160 [-0.0422, 0.0628] | 0.0135 [-0.0185, 0.0664] |
| R6 Stage3-Stage2 | 0.0376 [-0.0274, 0.1031] | 0.0515 [-0.0005, 0.1161] |
| R6 Stage4-Stage3 | 0.0297 [-0.0248, 0.0789] | 0.0254 [-0.0740, 0.1348] |
| R6 Stage4-Stage1 | 0.0833 [0.0135, 0.1410] | 0.0905 [0.0003, 0.2146] |
| R6−R5 Stage 1 | 0.0118 [-0.0135, 0.0370] | 0.0018 [-0.0082, 0.0203] |
| R6−R5 Stage 2 | -0.0033 [-0.0469, 0.0317] | 0.0102 [-0.0089, 0.0460] |
| R6−R5 Stage 3 | 0.0013 [-0.0282, 0.0340] | 0.0051 [-0.0425, 0.0659] |
| R6−R5 Stage 4 | -0.0276 [-0.0743, 0.0126] | 0.0156 [-0.0997, 0.1526] |

## Best Stage, joint probes, and PCA-64

| Backbone | Best Stage | Paired best−second AUROC (95% CI) | Advantage |
|---|---|---:|---|
| R5 | Stage 4 | 0.0586 [0.0044, 0.1224] | Clear in this paired interval |
| R6 | Stage 4 | 0.0297 [-0.0248, 0.0789] | No clear stage advantage |
| R5 Stage3+4−Stage3 | 0.0724 [0.0237, 0.1271] | 0.0257 [-0.0248, 0.0780] |
| R5 Stage3+4−Stage4 | 0.0138 [-0.0063, 0.0343] | 0.0107 [-0.0526, 0.0307] |
| R5 Stage2+3−Stage2 | 0.0428 [-0.0111, 0.0930] | 0.0586 [0.0065, 0.0931] |
| R5 Stage2+3−Stage3 | 0.0098 [-0.0036, 0.0283] | 0.0020 [-0.0219, 0.0176] |
| R6 Stage3+4−Stage3 | 0.0566 [0.0073, 0.0994] | 0.0196 [-0.0631, 0.1188] |
| R6 Stage3+4−Stage4 | 0.0268 [0.0094, 0.0477] | -0.0059 [-0.0409, 0.0275] |
| R6 Stage2+3−Stage2 | 0.0393 [-0.0234, 0.1037] | 0.0501 [0.0021, 0.1073] |
| R6 Stage2+3−Stage3 | 0.0017 [-0.0127, 0.0149] | -0.0014 [-0.0160, 0.0098] |

## Disease-specific evidence and next direction

- R5: point-best Stage 4; Stage3+4−Stage4 AUROC +0.0138 [-0.0063, +0.0343]. Stage3+Stage4 has a positive point change over Stage4, but its interval includes zero; no structural direction is confirmed.
- R6: point-best Stage 4; Stage3+4−Stage4 AUROC +0.0268 [+0.0094, +0.0477]. Stage3+Stage4 provides positive paired evidence beyond Stage4 under this probe; prioritize a controlled Stage3→4 feature-transfer study.


PCA64 rows show the PCA-64 probe minus the corresponding raw Stage probe. PCA and scaling are fitted only on the current Train fold during CV, then on all Train for final Validation prediction.

| Backbone | Stage | Δ AUROC (95% CI) | Δ AUPRC (95% CI) |
|---|---|---:|---:|
| R5 | Stage 1 | -0.0061 [-0.0118, -0.0008] | -0.0024 [-0.0062, 0.0001] |
| R5 | Stage 2 | -0.0053 [-0.0254, 0.0159] | -0.0032 [-0.0193, 0.0026] |
| R5 | Stage 3 | 0.0239 [-0.0064, 0.0549] | 0.0086 [-0.0178, 0.0338] |
| R5 | Stage 4 | 0.0132 [-0.0210, 0.0403] | 0.0023 [-0.0760, 0.0382] |
| R6 | Stage 1 | -0.0044 [-0.0139, 0.0021] | -0.0017 [-0.0086, 0.0051] |
| R6 | Stage 2 | 0.0003 [-0.0171, 0.0184] | -0.0043 [-0.0232, 0.0073] |
| R6 | Stage 3 | -0.0112 [-0.0469, 0.0145] | 0.0073 [-0.0674, 0.0960] |
| R6 | Stage 4 | 0.0206 [-0.0059, 0.0468] | -0.0186 [-0.1119, 0.0260] |

## Interpretation limits

These are Study-level linear probes on frozen features under deterministic ResizePad+GAP and the specified L2 Logistic Regression protocol. A higher Stage score is evidence of easier linear separability under this probe, not proof that a Stage contains uniquely important pathology information. Confidence intervals are exploratory because of the many disease/Stage/probe comparisons and repeated Validation use for historical checkpoint selection. No disease localization or causal claims are made.
