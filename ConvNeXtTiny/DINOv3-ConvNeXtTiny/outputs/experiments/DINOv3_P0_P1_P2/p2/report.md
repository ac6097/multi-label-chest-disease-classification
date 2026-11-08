# DINOv3 P2 Training Report

- Status: 8/20 epochs; stop reason: `three_condition_early_stop`.
- Best epoch: 5; Macro-AUROC: 0.794882; Valid BCE: 0.311232.
- Last epoch: 8; Train BCE: 0.191905; Valid BCE: 0.394731; Macro-AUROC: 0.766137.
- Trainable parameters: 26,592,778; frozen parameters: 1,235,040.
- Actual mean train throughput: 23.58 images/s.
- Best checkpoint: `best.pth`; last checkpoint: `last.pth`.
- Test split was not read.

| Epoch | Train BCE | Valid BCE | Macro-AUROC | Macro-AUPRC | Macro-F1 | images/s | Train down 3 | Valid up 3 | AUROC no improve 3 | Stop |
|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| 1 | 0.372122 | 0.363073 | 0.724890 | 0.332009 | 0.176385 | 22.83 | False | False | False | False |
| 2 | 0.328494 | 0.329307 | 0.770718 | 0.377533 | 0.226606 | 23.98 | False | False | False | False |
| 3 | 0.312707 | 0.322789 | 0.786419 | 0.405509 | 0.282366 | 24.83 | False | False | False | False |
| 4 | 0.298019 | 0.342382 | 0.786337 | 0.408278 | 0.251724 | 24.41 | True | False | False | False |
| 5 | 0.282861 | 0.311232 | 0.794882 | 0.415215 | 0.265375 | 23.00 | True | False | False | False |
| 6 | 0.260615 | 0.326617 | 0.788458 | 0.410275 | 0.325616 | 22.96 | True | False | False | False |
| 7 | 0.230886 | 0.343468 | 0.779203 | 0.399296 | 0.320090 | 22.90 | True | False | False | False |
| 8 | 0.191905 | 0.394731 | 0.766137 | 0.375021 | 0.317027 | 23.70 | True | True | True | True |
