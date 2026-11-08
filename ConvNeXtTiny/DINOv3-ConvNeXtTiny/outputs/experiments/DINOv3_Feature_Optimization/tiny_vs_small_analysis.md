# R4 DINOv3 ConvNeXt-Small vs matched Tiny GAP

- Candidate: `R4`; reference: `R2`.
- Macro-AUROC: 0.799287 vs 0.802116.
- Difference: -0.002829; subject-bootstrap 95% CI [-0.013780, +0.009827].
- Paired validation: 859 Studies / 228 Subjects; 2000 valid bootstrap replicates.
- Best epoch selection is based on this same Validation set; intervals are conditional and do not replace an independent Test evaluation.
- Test split was not read.

## Additional measurements

- Small Best Epoch: 6; stop epoch: 9 (`three_condition_early_stop`).
- Small initialization SHA-256: `8182492b73345ceb449ed6be0de486d778a73afb273ba2684de785a71ef1da9d`.
- Trainable parameters: 48,227,338; total: 49,462,378.
- Mean formal train throughput: 24.17 images/s.
- Resolution: 512.

## Per-class AUROC differences

| Class | Delta | 95% CI |
|---|---:|---:|
| Enlarged Cardiomediastinum | -0.010304 | [-0.079313, +0.088287] |
| Pneumothorax | -0.006888 | [-0.041033, +0.024164] |
| Consolidation | +0.011030 | [-0.017797, +0.048090] |
| Pneumonia | -0.035500 | [-0.086500, +0.012853] |
| Edema | -0.003144 | [-0.021439, +0.015244] |
| Cardiomegaly | +0.004051 | [-0.016903, +0.025951] |
| Atelectasis | +0.012965 | [-0.004808, +0.033560] |
| Lung Opacity | +0.008027 | [-0.014135, +0.027897] |
| Pleural Effusion | -0.002661 | [-0.013297, +0.007865] |
| No Finding | -0.005868 | [-0.021942, +0.010710] |
