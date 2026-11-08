# R7 training report

- Backbone: DINOv3 ConvNeXt-Tiny R7 Stage3→Stage4 Residual Transfer; input: 512×512; head: GAP + Linear(10).
- Stop: epoch 9/20 (`three_condition_early_stop`).
- Best Epoch: 6; Macro-AUROC: 0.802901.
- Best Train BCE / Valid BCE: 0.262602 / 0.311513.
- Last Train BCE / Valid BCE / Macro-AUROC: 0.142555 / 0.458949 / 0.773139.
- Mean formal train throughput: 20.35 images/s.
- Mean DataLoader wait: 1147.9 seconds/epoch.
- Trainable / frozen parameters: 26,887,691 / 1,235,040.
- Study-level BCE uses the arithmetic mean of raw image logits within each Study.
- Test split was not read.

| Epoch | Train BCE | Valid BCE | Macro-AUROC | Macro-AUPRC | Macro-F1 | images/s | Train down 3 | Valid up 3 | AUROC no improve 3 | Stop |
|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| 1 | 0.371821 | 0.343340 | 0.733684 | 0.342318 | 0.160210 | 21.33 | False | False | False | False |
| 2 | 0.327374 | 0.330690 | 0.769518 | 0.375325 | 0.209668 | 19.21 | False | False | False | False |
| 3 | 0.312353 | 0.321028 | 0.786028 | 0.400141 | 0.283612 | 18.23 | False | False | False | False |
| 4 | 0.299403 | 0.346179 | 0.786510 | 0.409328 | 0.238916 | 20.06 | True | False | False | False |
| 5 | 0.284640 | 0.310041 | 0.797427 | 0.414137 | 0.251340 | 22.08 | True | False | False | False |
| 6 | 0.262602 | 0.311513 | 0.802901 | 0.427874 | 0.317050 | 18.41 | True | False | False | False |
| 7 | 0.231017 | 0.339829 | 0.795820 | 0.412669 | 0.327528 | 21.08 | True | False | False | False |
| 8 | 0.188992 | 0.398587 | 0.788072 | 0.409085 | 0.346118 | 20.70 | True | True | False | False |
| 9 | 0.142555 | 0.458949 | 0.773139 | 0.390069 | 0.364908 | 22.06 | True | True | True | True |

## R7 residual branch

R7 retains the R5 Stage3/Stage4 freeze range and original Global Head, adding only AvgPool2d(2,2) + Conv2d(384,768,1,bias=False) with scalar gamma initialized at zero. Per-epoch branch activity is in `r7_branch_diagnostics.csv`.
