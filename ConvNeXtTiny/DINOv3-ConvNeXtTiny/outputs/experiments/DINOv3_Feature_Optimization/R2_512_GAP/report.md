# DINOv3 P2 Training Report

- Status: 9/20 epochs; stop reason: `three_condition_early_stop`.
- Best epoch: 6; Macro-AUROC: 0.802116; Valid BCE: 0.311968.
- Last epoch: 9; Train BCE: 0.149424; Valid BCE: 0.440514; Macro-AUROC: 0.769639.
- Trainable parameters: 26,592,778; frozen parameters: 1,235,040.
- Actual mean train throughput: 22.99 images/s.
- Best checkpoint: `best.pth`; last checkpoint: `last.pth`.
- Test split was not read.

| Epoch | Train BCE | Valid BCE | Macro-AUROC | Macro-AUPRC | Macro-F1 | images/s | Train down 3 | Valid up 3 | AUROC no improve 3 | Stop |
|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| 1 | 0.372739 | 0.358549 | 0.730838 | 0.342163 | 0.172498 | 22.75 | False | False | False | False |
| 2 | 0.327911 | 0.329765 | 0.765148 | 0.370210 | 0.236743 | 22.83 | False | False | False | False |
| 3 | 0.313061 | 0.320257 | 0.785601 | 0.403996 | 0.249825 | 23.22 | False | False | False | False |
| 4 | 0.299831 | 0.339981 | 0.789042 | 0.411275 | 0.237673 | 23.21 | True | False | False | False |
| 5 | 0.284962 | 0.311995 | 0.794396 | 0.412395 | 0.254121 | 23.01 | True | False | False | False |
| 6 | 0.264244 | 0.311968 | 0.802116 | 0.421206 | 0.310561 | 22.93 | True | False | False | False |
| 7 | 0.234526 | 0.332891 | 0.794719 | 0.409520 | 0.300574 | 22.95 | True | False | False | False |
| 8 | 0.193799 | 0.392030 | 0.779840 | 0.391572 | 0.326545 | 23.02 | True | False | False | False |
| 9 | 0.149424 | 0.440514 | 0.769639 | 0.384795 | 0.347225 | 22.97 | True | True | True | True |
