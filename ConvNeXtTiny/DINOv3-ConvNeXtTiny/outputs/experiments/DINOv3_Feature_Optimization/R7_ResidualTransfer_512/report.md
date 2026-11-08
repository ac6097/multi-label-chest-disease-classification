# DINOv3 P2 Training Report

- Status: 9/20 epochs; stop reason: `three_condition_early_stop`.
- Best epoch: 6; Macro-AUROC: 0.802901; Valid BCE: 0.311513.
- Last epoch: 9; Train BCE: 0.142555; Valid BCE: 0.458949; Macro-AUROC: 0.773139.
- Trainable parameters: 26,887,691; frozen parameters: 1,235,040.
- Actual mean train throughput: 20.35 images/s.
- Best checkpoint: `best.pth`; last checkpoint: `last.pth`.
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
