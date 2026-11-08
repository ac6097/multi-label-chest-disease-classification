# DINOv3 P2 Training Report

- Status: 9/20 epochs; stop reason: `three_condition_early_stop`.
- Best epoch: 6; Macro-AUROC: 0.802962; Valid BCE: 0.311438.
- Last epoch: 9; Train BCE: 0.142414; Valid BCE: 0.454692; Macro-AUROC: 0.775063.
- Trainable parameters: 26,694,686; frozen parameters: 1,235,040.
- Actual mean train throughput: 22.21 images/s.
- Best checkpoint: `best.pth`; last checkpoint: `last.pth`.
- Test split was not read.

| Epoch | Train BCE | Valid BCE | Macro-AUROC | Macro-AUPRC | Macro-F1 | images/s | Train down 3 | Valid up 3 | AUROC no improve 3 | Stop |
|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| 1 | 0.371808 | 0.343246 | 0.733719 | 0.342891 | 0.159950 | 22.88 | False | False | False | False |
| 2 | 0.327335 | 0.330561 | 0.769652 | 0.376218 | 0.209089 | 20.42 | False | False | False | False |
| 3 | 0.312303 | 0.321026 | 0.785966 | 0.400786 | 0.283486 | 21.85 | False | False | False | False |
| 4 | 0.299382 | 0.346180 | 0.786242 | 0.409655 | 0.237449 | 23.11 | True | False | False | False |
| 5 | 0.284601 | 0.310010 | 0.797365 | 0.414083 | 0.250875 | 22.57 | True | False | False | False |
| 6 | 0.262541 | 0.311438 | 0.802962 | 0.427697 | 0.312492 | 22.20 | True | False | False | False |
| 7 | 0.230857 | 0.339013 | 0.796069 | 0.414597 | 0.324896 | 22.35 | True | False | False | False |
| 8 | 0.188889 | 0.394509 | 0.788278 | 0.411529 | 0.347706 | 22.18 | True | True | False | False |
| 9 | 0.142414 | 0.454692 | 0.775063 | 0.391847 | 0.365551 | 22.33 | True | True | True | True |
