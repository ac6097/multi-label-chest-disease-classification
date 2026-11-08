# DINOv3 P2 Training Report

- Status: 9/20 epochs; stop reason: `three_condition_early_stop`.
- Best epoch: 6; Macro-AUROC: 0.799287; Valid BCE: 0.321646.
- Last epoch: 9; Train BCE: 0.130424; Valid BCE: 0.479786; Macro-AUROC: 0.771064.
- Trainable parameters: 48,227,338; frozen parameters: 1,235,040.
- Actual mean train throughput: 24.17 images/s.
- Best checkpoint: `best.pth`; last checkpoint: `last.pth`.
- Test split was not read.

| Epoch | Train BCE | Valid BCE | Macro-AUROC | Macro-AUPRC | Macro-F1 | images/s | Train down 3 | Valid up 3 | AUROC no improve 3 | Stop |
|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| 1 | 0.400062 | 0.364574 | 0.730217 | 0.338166 | 0.183247 | 22.73 | False | False | False | False |
| 2 | 0.327014 | 0.326096 | 0.771048 | 0.379616 | 0.249697 | 24.30 | False | False | False | False |
| 3 | 0.310813 | 0.317982 | 0.789285 | 0.411531 | 0.278999 | 24.34 | False | False | False | False |
| 4 | 0.294668 | 0.351422 | 0.779085 | 0.406046 | 0.224662 | 24.30 | True | False | False | False |
| 5 | 0.276688 | 0.310198 | 0.799087 | 0.422464 | 0.289882 | 24.46 | True | False | False | False |
| 6 | 0.248480 | 0.321646 | 0.799287 | 0.413240 | 0.335200 | 24.06 | True | False | False | False |
| 7 | 0.212711 | 0.347198 | 0.789219 | 0.409156 | 0.334699 | 24.19 | True | False | False | False |
| 8 | 0.172185 | 0.418253 | 0.785620 | 0.402521 | 0.388472 | 24.45 | True | True | False | False |
| 9 | 0.130424 | 0.479786 | 0.771064 | 0.389060 | 0.384729 | 24.72 | True | True | True | True |
