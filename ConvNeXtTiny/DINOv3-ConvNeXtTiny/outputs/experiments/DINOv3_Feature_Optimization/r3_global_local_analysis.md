# R3 Global GAP plus Local Disease Response

- Candidate: `R3`; reference: `R2`.
- Macro-AUROC: 0.804653 vs 0.802116.
- Difference: +0.002537; subject-bootstrap 95% CI [-0.004813, +0.009448].
- Paired validation: 859 Studies / 228 Subjects; 2000 valid bootstrap replicates.
- Best epoch selection is based on this same Validation set; intervals are conditional and do not replace an independent Test evaluation.
- Test split was not read.

## Additional measurements

- Best Epoch: 6; stop epoch: 9 (`three_condition_early_stop`).
- Final learned alpha: 0.050662; initial alpha: 0.050000; bound: [0, 0.20].
- Top-K ratio: 0.10; selected resolution: 512.
- Additional parameters versus GAP: 3,851.
- Mean formal train throughput: 21.45 images/s.
- Validation seconds at best epoch: 112.67.
- Global classifier initialization SHA-256 matches the R0 seed: 40dc9552892097293310d9fca16e61913c5fa11eae96e3d56e572553141827a9.

## Per-class AUROC differences

| Class | Delta | 95% CI |
|---|---:|---:|
| Enlarged Cardiomediastinum | +0.022056 | [-0.016111, +0.070177] |
| Pneumothorax | -0.000066 | [-0.020899, +0.017132] |
| Consolidation | -0.004920 | [-0.041610, +0.018974] |
| Pneumonia | -0.006832 | [-0.045216, +0.033138] |
| Edema | -0.000473 | [-0.009304, +0.006980] |
| Cardiomegaly | -0.002092 | [-0.020920, +0.017568] |
| Atelectasis | +0.011467 | [-0.006271, +0.030057] |
| Lung Opacity | +0.005879 | [-0.010596, +0.020467] |
| Pleural Effusion | +0.000918 | [-0.003935, +0.005894] |
| No Finding | -0.000567 | [-0.009245, +0.007705] |
