# Pneumonia

- Validation positive / negative Studies: 68 / 791; Subjects: 228.

## Single-stage probe

| Backbone | Stage | AUROC (95% CI) | AUPRC (95% CI) | OOF AUROC | OOF AUPRC | C | Dim | F1@0.5 | Balanced Acc@0.5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R5 | Stage 1 | 0.5854 [0.5175, 0.6693] | 0.1044 [0.0746, 0.1660] | 0.6074 | 0.1052 | 0.01 | 96 | 0.0000 | 0.5000 |
| R5 | Stage 2 | 0.5888 [0.5192, 0.6658] | 0.1017 [0.0715, 0.1596] | 0.6219 | 0.1156 | 0.1 | 192 | 0.0000 | 0.5000 |
| R5 | Stage 3 | 0.6782 [0.6117, 0.7526] | 0.1732 [0.1162, 0.2522] | 0.7299 | 0.1757 | 0.01 | 384 | 0.0000 | 0.5000 |
| R5 | Stage 4 | 0.6725 [0.5934, 0.7459] | 0.1683 [0.1120, 0.2534] | 0.8657 | 0.4029 | 0.01 | 768 | 0.0988 | 0.5237 |
| R6 | Stage 1 | 0.5955 [0.5256, 0.6690] | 0.0989 [0.0675, 0.1457] | 0.6045 | 0.1092 | 10 | 96 | 0.0000 | 0.5000 |
| R6 | Stage 2 | 0.6399 [0.5739, 0.7055] | 0.1323 [0.0926, 0.1929] | 0.6345 | 0.1193 | 0.01 | 192 | 0.0000 | 0.5000 |
| R6 | Stage 3 | 0.6675 [0.5898, 0.7391] | 0.1714 [0.1145, 0.2682] | 0.7358 | 0.1817 | 0.01 | 384 | 0.0000 | 0.5000 |
| R6 | Stage 4 | 0.6921 [0.6245, 0.7590] | 0.1729 [0.1215, 0.2768] | 0.8536 | 0.3714 | 0.01 | 768 | 0.0519 | 0.5103 |

## Stage transitions and R6−R5

| Comparison | Δ AUROC (95% CI) | Δ AUPRC (95% CI) |
|---|---:|---:|
| R5 Stage2-Stage1 | 0.0033 [-0.0546, 0.0533] | -0.0027 [-0.0367, 0.0273] |
| R5 Stage3-Stage2 | 0.0894 [0.0137, 0.1660] | 0.0715 [0.0088, 0.1403] |
| R5 Stage4-Stage3 | -0.0057 [-0.0712, 0.0549] | -0.0048 [-0.0709, 0.0707] |
| R5 Stage4-Stage1 | 0.0871 [-0.0155, 0.1709] | 0.0639 [-0.0138, 0.1456] |
| R6 Stage2-Stage1 | 0.0444 [-0.0099, 0.0990] | 0.0334 [0.0049, 0.0757] |
| R6 Stage3-Stage2 | 0.0276 [-0.0549, 0.1080] | 0.0392 [-0.0265, 0.1312] |
| R6 Stage4-Stage3 | 0.0246 [-0.0286, 0.0795] | 0.0014 [-0.0735, 0.0896] |
| R6 Stage4-Stage1 | 0.0966 [0.0306, 0.1577] | 0.0740 [0.0301, 0.1664] |
| R6−R5 Stage 1 | 0.0101 [-0.0470, 0.0572] | -0.0055 [-0.0427, 0.0147] |
| R6−R5 Stage 2 | 0.0511 [-0.0047, 0.0964] | 0.0306 [-0.0043, 0.0676] |
| R6−R5 Stage 3 | -0.0107 [-0.0608, 0.0363] | -0.0017 [-0.0591, 0.0631] |
| R6−R5 Stage 4 | 0.0196 [-0.0374, 0.0724] | 0.0045 [-0.0544, 0.0909] |

## Best Stage, joint probes, and PCA-64

| Backbone | Best Stage | Paired best−second AUROC (95% CI) | Advantage |
|---|---|---:|---|
| R5 | Stage 3 | 0.0057 [-0.0549, 0.0712] | No clear stage advantage |
| R6 | Stage 4 | 0.0246 [-0.0286, 0.0795] | No clear stage advantage |
| R5 Stage3+4−Stage3 | 0.0212 [-0.0271, 0.0676] | 0.0080 [-0.0480, 0.0819] |
| R5 Stage3+4−Stage4 | 0.0269 [0.0042, 0.0532] | 0.0128 [-0.0039, 0.0447] |
| R5 Stage2+3−Stage2 | 0.0781 [0.0063, 0.1454] | 0.0518 [0.0023, 0.1024] |
| R5 Stage2+3−Stage3 | -0.0113 [-0.0340, 0.0088] | -0.0197 [-0.0484, 0.0035] |
| R6 Stage3+4−Stage3 | 0.0241 [-0.0229, 0.0708] | 0.0008 [-0.0626, 0.0976] |
| R6 Stage3+4−Stage4 | -0.0004 [-0.0206, 0.0177] | -0.0007 [-0.0293, 0.0482] |
| R6 Stage2+3−Stage2 | 0.0367 [-0.0458, 0.1169] | 0.0394 [-0.0241, 0.1250] |
| R6 Stage2+3−Stage3 | 0.0091 [-0.0088, 0.0245] | 0.0002 [-0.0223, 0.0178] |

## Disease-specific evidence and next direction

- R5: point-best Stage 3; Stage3+4−Stage4 AUROC +0.0269 [+0.0042, +0.0532]. Stage3+Stage4 provides positive paired evidence beyond Stage4 under this probe; prioritize a controlled Stage3→4 feature-transfer study.
- R6: point-best Stage 4; Stage3+4−Stage4 AUROC -0.0004 [-0.0206, +0.0177]. Current stage and joint-probe intervals do not isolate a clear bottleneck; retain this disease as an open question for a pre-registered follow-up.


PCA64 rows show the PCA-64 probe minus the corresponding raw Stage probe. PCA and scaling are fitted only on the current Train fold during CV, then on all Train for final Validation prediction.

| Backbone | Stage | Δ AUROC (95% CI) | Δ AUPRC (95% CI) |
|---|---|---:|---:|
| R5 | Stage 1 | 0.0000 [-0.0026, 0.0022] | -0.0004 [-0.0021, 0.0007] |
| R5 | Stage 2 | -0.0058 [-0.0524, 0.0514] | -0.0055 [-0.0330, 0.0095] |
| R5 | Stage 3 | 0.0234 [-0.0155, 0.0558] | 0.0043 [-0.0354, 0.0462] |
| R5 | Stage 4 | 0.0226 [-0.0110, 0.0571] | 0.0064 [-0.0314, 0.0490] |
| R6 | Stage 1 | -0.0056 [-0.0629, 0.0595] | 0.0051 [-0.0174, 0.0390] |
| R6 | Stage 2 | -0.0234 [-0.0516, 0.0158] | -0.0182 [-0.0405, -0.0003] |
| R6 | Stage 3 | 0.0334 [-0.0027, 0.0696] | -0.0018 [-0.0583, 0.0558] |
| R6 | Stage 4 | -0.0108 [-0.0433, 0.0156] | -0.0175 [-0.0764, 0.0261] |

## Interpretation limits

These are Study-level linear probes on frozen features under deterministic ResizePad+GAP and the specified L2 Logistic Regression protocol. A higher Stage score is evidence of easier linear separability under this probe, not proof that a Stage contains uniquely important pathology information. Confidence intervals are exploratory because of the many disease/Stage/probe comparisons and repeated Validation use for historical checkpoint selection. No disease localization or causal claims are made.
