# Edema

- Validation positive / negative Studies: 115 / 744; Subjects: 228.

## Single-stage probe

| Backbone | Stage | AUROC (95% CI) | AUPRC (95% CI) | OOF AUROC | OOF AUPRC | C | Dim | F1@0.5 | Balanced Acc@0.5 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R5 | Stage 1 | 0.8140 [0.7738, 0.8610] | 0.3329 [0.2707, 0.4181] | 0.8153 | 0.3505 | 100 | 96 | 0.1477 | 0.5324 |
| R5 | Stage 2 | 0.8264 [0.7784, 0.8781] | 0.3700 [0.2951, 0.5327] | 0.8295 | 0.3726 | 0.1 | 192 | 0.2323 | 0.5635 |
| R5 | Stage 3 | 0.8643 [0.8187, 0.9093] | 0.4203 [0.3142, 0.5891] | 0.9082 | 0.6073 | 0.01 | 384 | 0.4000 | 0.6437 |
| R5 | Stage 4 | 0.8789 [0.8373, 0.9222] | 0.5125 [0.3789, 0.6741] | 0.9380 | 0.7168 | 0.001 | 768 | 0.5091 | 0.7105 |
| R6 | Stage 1 | 0.8336 [0.7904, 0.8809] | 0.3562 [0.2858, 0.4669] | 0.8213 | 0.3601 | 1 | 96 | 0.2331 | 0.5631 |
| R6 | Stage 2 | 0.8239 [0.7773, 0.8745] | 0.3647 [0.2895, 0.4936] | 0.8393 | 0.3849 | 0.1 | 192 | 0.2614 | 0.5745 |
| R6 | Stage 3 | 0.8657 [0.8206, 0.9074] | 0.4088 [0.3027, 0.5997] | 0.9092 | 0.6046 | 0.01 | 384 | 0.3768 | 0.6339 |
| R6 | Stage 4 | 0.8594 [0.8153, 0.9026] | 0.4815 [0.3529, 0.6212] | 0.9371 | 0.7136 | 0.01 | 768 | 0.4404 | 0.6717 |

## Stage transitions and R6−R5

| Comparison | Δ AUROC (95% CI) | Δ AUPRC (95% CI) |
|---|---:|---:|
| R5 Stage2-Stage1 | 0.0124 [-0.0166, 0.0454] | 0.0370 [-0.0138, 0.1520] |
| R5 Stage3-Stage2 | 0.0378 [0.0018, 0.0720] | 0.0504 [-0.0727, 0.1694] |
| R5 Stage4-Stage3 | 0.0146 [-0.0024, 0.0311] | 0.0922 [0.0262, 0.1497] |
| R5 Stage4-Stage1 | 0.0649 [0.0253, 0.1032] | 0.1796 [0.0734, 0.3119] |
| R6 Stage2-Stage1 | -0.0098 [-0.0432, 0.0220] | 0.0084 [-0.0637, 0.0792] |
| R6 Stage3-Stage2 | 0.0418 [0.0074, 0.0786] | 0.0442 [-0.0208, 0.1668] |
| R6 Stage4-Stage3 | -0.0063 [-0.0294, 0.0146] | 0.0726 [-0.0258, 0.1196] |
| R6 Stage4-Stage1 | 0.0257 [-0.0127, 0.0580] | 0.1252 [0.0243, 0.1985] |
| R6−R5 Stage 1 | 0.0196 [0.0025, 0.0361] | 0.0233 [-0.0068, 0.0855] |
| R6−R5 Stage 2 | -0.0026 [-0.0175, 0.0112] | -0.0053 [-0.0615, 0.0356] |
| R6−R5 Stage 3 | 0.0014 [-0.0176, 0.0187] | -0.0115 [-0.0583, 0.0577] |
| R6−R5 Stage 4 | -0.0195 [-0.0372, -0.0048] | -0.0311 [-0.0909, 0.0075] |

## Best Stage, joint probes, and PCA-64

| Backbone | Best Stage | Paired best−second AUROC (95% CI) | Advantage |
|---|---|---:|---|
| R5 | Stage 4 | 0.0146 [-0.0024, 0.0311] | No clear stage advantage |
| R6 | Stage 3 | 0.0063 [-0.0146, 0.0294] | No clear stage advantage |
| R5 Stage3+4−Stage3 | 0.0134 [-0.0032, 0.0299] | 0.0982 [0.0307, 0.1500] |
| R5 Stage3+4−Stage4 | -0.0012 [-0.0046, 0.0024] | 0.0060 [-0.0078, 0.0181] |
| R5 Stage2+3−Stage2 | 0.0410 [0.0031, 0.0764] | 0.0607 [-0.0660, 0.1778] |
| R5 Stage2+3−Stage3 | 0.0032 [-0.0009, 0.0076] | 0.0104 [-0.0055, 0.0241] |
| R6 Stage3+4−Stage3 | 0.0003 [-0.0164, 0.0162] | 0.0641 [-0.0034, 0.1031] |
| R6 Stage3+4−Stage4 | 0.0066 [-0.0009, 0.0149] | -0.0085 [-0.0340, 0.0374] |
| R6 Stage2+3−Stage2 | 0.0430 [0.0077, 0.0818] | 0.0437 [-0.0219, 0.1622] |
| R6 Stage2+3−Stage3 | 0.0012 [-0.0024, 0.0054] | -0.0005 [-0.0180, 0.0125] |

## Disease-specific evidence and next direction

- R5: point-best Stage 4; Stage3+4−Stage4 AUROC -0.0012 [-0.0046, +0.0024]. Current stage and joint-probe intervals do not isolate a clear bottleneck; retain this disease as an open question for a pre-registered follow-up.
- R6: point-best Stage 3; Stage3+4−Stage4 AUROC +0.0066 [-0.0009, +0.0149]. Stage3+Stage4 has a positive point change over Stage4, but its interval includes zero; no structural direction is confirmed.


PCA64 rows show the PCA-64 probe minus the corresponding raw Stage probe. PCA and scaling are fitted only on the current Train fold during CV, then on all Train for final Validation prediction.

| Backbone | Stage | Δ AUROC (95% CI) | Δ AUPRC (95% CI) |
|---|---|---:|---:|
| R5 | Stage 1 | -0.0104 [-0.0216, -0.0000] | -0.0024 [-0.0282, 0.0347] |
| R5 | Stage 2 | -0.0172 [-0.0353, 0.0070] | -0.0314 [-0.1307, 0.0137] |
| R5 | Stage 3 | 0.0017 [-0.0088, 0.0131] | 0.0095 [-0.0171, 0.0525] |
| R5 | Stage 4 | -0.0003 [-0.0055, 0.0033] | -0.0107 [-0.0230, 0.0129] |
| R6 | Stage 1 | -0.0071 [-0.0170, 0.0020] | -0.0205 [-0.0605, -0.0025] |
| R6 | Stage 2 | 0.0011 [-0.0115, 0.0175] | -0.0089 [-0.0463, 0.0240] |
| R6 | Stage 3 | 0.0012 [-0.0089, 0.0136] | 0.0184 [-0.0218, 0.0583] |
| R6 | Stage 4 | 0.0079 [-0.0016, 0.0196] | -0.0048 [-0.0369, 0.0622] |

## Interpretation limits

These are Study-level linear probes on frozen features under deterministic ResizePad+GAP and the specified L2 Logistic Regression protocol. A higher Stage score is evidence of easier linear separability under this probe, not proof that a Stage contains uniquely important pathology information. Confidence intervals are exploratory because of the many disease/Stage/probe comparisons and repeated Validation use for historical checkpoint selection. No disease localization or causal claims are made.
