# DINOv3 P1 Training Report

- Status: 8/20 epochs; stop reason: `three_condition_early_stop`.
- Best epoch: 5; Macro-AUROC: 0.766191; Valid BCE: 0.324436.
- Last epoch: 8; Train BCE: 0.235868; Valid BCE: 0.356191; Macro-AUROC: 0.760476.
- Trainable parameters: 15,479,818; frozen parameters: 12,348,000.
- Actual mean train throughput: 23.19 images/s.
- Best checkpoint: `best.pth`; last checkpoint: `last.pth`.
- Test split was not read.

| Epoch | Train BCE | Valid BCE | Macro-AUROC | Macro-AUPRC | Macro-F1 | images/s | Train down 3 | Valid up 3 | AUROC no improve 3 | Stop |
|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| 1 | 0.386747 | 0.355504 | 0.708025 | 0.302776 | 0.106498 | 23.15 | False | False | False | False |
| 2 | 0.338727 | 0.348310 | 0.737149 | 0.336862 | 0.168029 | 23.14 | False | False | False | False |
| 3 | 0.324814 | 0.333798 | 0.758739 | 0.363448 | 0.214337 | 23.15 | False | False | False | False |
| 4 | 0.312593 | 0.346426 | 0.759595 | 0.370032 | 0.199839 | 22.47 | True | False | False | False |
| 5 | 0.301960 | 0.324436 | 0.766191 | 0.376917 | 0.209051 | 23.15 | True | False | False | False |
| 6 | 0.286282 | 0.328568 | 0.765887 | 0.379308 | 0.263596 | 23.37 | True | False | False | False |
| 7 | 0.264847 | 0.345062 | 0.752524 | 0.376820 | 0.266624 | 23.70 | True | False | False | False |
| 8 | 0.235868 | 0.356191 | 0.760476 | 0.389656 | 0.294522 | 23.37 | True | True | True | True |
