# No Finding

- Validation positive / negative Studies: 306 / 553; Subjects: 228.

## Single-stage probe

| Backbone | Stage | AUROC (95% CI) | AUPRC (95% CI) | OOF AUROC | OOF AUPRC | C | Dim | F1@0.5 | Balanced Acc@0.5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R5 | Stage 1 | 0.8082 [0.7636, 0.8510] | 0.7419 [0.6677, 0.8131] | 0.8243 | 0.7483 | 10 | 96 | 0.6596 | 0.7388 |
| R5 | Stage 2 | 0.8256 [0.7821, 0.8659] | 0.7647 [0.6921, 0.8317] | 0.8361 | 0.7597 | 0.1 | 192 | 0.6829 | 0.7552 |
| R5 | Stage 3 | 0.8562 [0.8184, 0.8910] | 0.8045 [0.7388, 0.8619] | 0.9223 | 0.8876 | 0.1 | 384 | 0.7079 | 0.7733 |
| R5 | Stage 4 | 0.8474 [0.8106, 0.8820] | 0.7942 [0.7306, 0.8539] | 0.9677 | 0.9531 | 0.01 | 768 | 0.7304 | 0.7900 |
| R6 | Stage 1 | 0.8166 [0.7731, 0.8570] | 0.7551 [0.6832, 0.8230] | 0.8261 | 0.7526 | 10 | 96 | 0.6581 | 0.7390 |
| R6 | Stage 2 | 0.8410 [0.8008, 0.8793] | 0.7758 [0.7054, 0.8442] | 0.8440 | 0.7724 | 1 | 192 | 0.7028 | 0.7697 |
| R6 | Stage 3 | 0.8504 [0.8094, 0.8894] | 0.7924 [0.7266, 0.8522] | 0.9241 | 0.8914 | 0.01 | 384 | 0.7194 | 0.7818 |
| R6 | Stage 4 | 0.8299 [0.7897, 0.8690] | 0.7689 [0.6992, 0.8343] | 0.9614 | 0.9434 | 0.01 | 768 | 0.7051 | 0.7713 |

## Stage transitions and R6−R5

| Comparison | Δ AUROC (95% CI) | Δ AUPRC (95% CI) |
|---|---:|---:|
| R5 Stage2-Stage1 | 0.0174 [0.0024, 0.0331] | 0.0228 [0.0008, 0.0465] |
| R5 Stage3-Stage2 | 0.0306 [0.0093, 0.0511] | 0.0398 [0.0130, 0.0666] |
| R5 Stage4-Stage3 | -0.0088 [-0.0266, 0.0086] | -0.0102 [-0.0323, 0.0141] |
| R5 Stage4-Stage1 | 0.0392 [0.0091, 0.0697] | 0.0523 [0.0173, 0.0908] |
| R6 Stage2-Stage1 | 0.0244 [0.0083, 0.0409] | 0.0207 [-0.0148, 0.0530] |
| R6 Stage3-Stage2 | 0.0094 [-0.0089, 0.0280] | 0.0166 [-0.0215, 0.0567] |
| R6 Stage4-Stage3 | -0.0205 [-0.0382, -0.0043] | -0.0235 [-0.0462, -0.0001] |
| R6 Stage4-Stage1 | 0.0133 [-0.0153, 0.0403] | 0.0138 [-0.0261, 0.0564] |
| R6−R5 Stage 1 | 0.0084 [-0.0016, 0.0188] | 0.0132 [-0.0040, 0.0314] |
| R6−R5 Stage 2 | 0.0154 [0.0036, 0.0277] | 0.0111 [-0.0165, 0.0382] |
| R6−R5 Stage 3 | -0.0058 [-0.0184, 0.0080] | -0.0121 [-0.0370, 0.0116] |
| R6−R5 Stage 4 | -0.0175 [-0.0371, 0.0022] | -0.0253 [-0.0556, 0.0033] |

## Best Stage, joint probes, and PCA-64

| Backbone | Best Stage | Paired best−second AUROC (95% CI) | Advantage |
|---|---|---:|---|
| R5 | Stage 3 | 0.0088 [-0.0086, 0.0266] | No clear stage advantage |
| R6 | Stage 3 | 0.0094 [-0.0089, 0.0280] | No clear stage advantage |
| R5 Stage3+4−Stage3 | -0.0108 [-0.0285, 0.0060] | -0.0124 [-0.0329, 0.0105] |
| R5 Stage3+4−Stage4 | -0.0020 [-0.0068, 0.0022] | -0.0021 [-0.0086, 0.0040] |
| R5 Stage2+3−Stage2 | 0.0376 [0.0168, 0.0582] | 0.0445 [0.0184, 0.0706] |
| R5 Stage2+3−Stage3 | 0.0070 [0.0022, 0.0117] | 0.0047 [-0.0038, 0.0133] |
| R6 Stage3+4−Stage3 | -0.0233 [-0.0428, -0.0060] | -0.0335 [-0.0562, -0.0096] |
| R6 Stage3+4−Stage4 | -0.0028 [-0.0086, 0.0033] | -0.0100 [-0.0178, -0.0017] |
| R6 Stage2+3−Stage2 | 0.0110 [-0.0077, 0.0291] | 0.0179 [-0.0212, 0.0578] |
| R6 Stage2+3−Stage3 | 0.0015 [-0.0017, 0.0045] | 0.0014 [-0.0032, 0.0056] |

## Disease-specific evidence and next direction

- R5: point-best Stage 3; Stage3+4−Stage4 AUROC -0.0020 [-0.0068, +0.0022]. Current stage and joint-probe intervals do not isolate a clear bottleneck; retain this disease as an open question for a pre-registered follow-up.
- R6: point-best Stage 3; Stage3+4−Stage4 AUROC -0.0028 [-0.0086, +0.0033]. Current stage and joint-probe intervals do not isolate a clear bottleneck; retain this disease as an open question for a pre-registered follow-up.


PCA64 rows show the PCA-64 probe minus the corresponding raw Stage probe. PCA and scaling are fitted only on the current Train fold during CV, then on all Train for final Validation prediction.

| Backbone | Stage | Δ AUROC (95% CI) | Δ AUPRC (95% CI) |
|---|---|---:|---:|
| R5 | Stage 1 | -0.0006 [-0.0079, 0.0068] | 0.0010 [-0.0093, 0.0121] |
| R5 | Stage 2 | -0.0129 [-0.0208, -0.0054] | -0.0251 [-0.0367, -0.0116] |
| R5 | Stage 3 | 0.0112 [-0.0027, 0.0252] | 0.0142 [-0.0052, 0.0349] |
| R5 | Stage 4 | 0.0045 [-0.0034, 0.0128] | -0.0003 [-0.0115, 0.0109] |
| R6 | Stage 1 | -0.0069 [-0.0169, 0.0028] | -0.0119 [-0.0326, 0.0063] |
| R6 | Stage 2 | -0.0199 [-0.0305, -0.0100] | -0.0227 [-0.0410, -0.0034] |
| R6 | Stage 3 | -0.0013 [-0.0095, 0.0065] | -0.0005 [-0.0134, 0.0122] |
| R6 | Stage 4 | 0.0039 [-0.0050, 0.0121] | 0.0004 [-0.0110, 0.0107] |

## Interpretation limits

These are Study-level linear probes on frozen features under deterministic ResizePad+GAP and the specified L2 Logistic Regression protocol. A higher Stage score is evidence of easier linear separability under this probe, not proof that a Stage contains uniquely important pathology information. Confidence intervals are exploratory because of the many disease/Stage/probe comparisons and repeated Validation use for historical checkpoint selection. No disease localization or causal claims are made.
