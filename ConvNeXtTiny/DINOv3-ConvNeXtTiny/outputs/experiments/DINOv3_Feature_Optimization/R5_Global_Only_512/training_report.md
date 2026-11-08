# R5_Global_Only_512 training report

- Backbone: DINOv3 ConvNeXt-Tiny; input: 512×512; head: GAP + Linear(10).
- Stop: epoch 9/20 (`three_condition_early_stop`).
- Best Epoch: 6; Macro-AUROC: 0.803514.
- Best Train BCE / Valid BCE: 0.262511 / 0.311168.
- Last Train BCE / Valid BCE / Macro-AUROC: 0.142537 / 0.453532 / 0.775818.
- Mean formal train throughput: 22.80 images/s.
- Mean DataLoader wait: 1003.0 seconds/epoch.
- Trainable / frozen parameters: 26,592,778 / 1,235,040.
- Study-level BCE uses the arithmetic mean of raw image logits within each Study.
- Test split was not read.

| Epoch | Train BCE | Valid BCE | Macro-AUROC | Macro-AUPRC | Macro-F1 | images/s | Train down 3 | Valid up 3 | AUROC no improve 3 | Stop |
|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| 1 | 0.371810 | 0.343373 | 0.733634 | 0.342581 | 0.160736 | 20.91 | False | False | False | False |
| 2 | 0.327348 | 0.330503 | 0.769780 | 0.375636 | 0.209812 | 21.87 | False | False | False | False |
| 3 | 0.312293 | 0.321029 | 0.785948 | 0.400576 | 0.283577 | 22.91 | False | False | False | False |
| 4 | 0.299369 | 0.345864 | 0.786444 | 0.409575 | 0.238928 | 22.52 | True | False | False | False |
| 5 | 0.284595 | 0.309922 | 0.797854 | 0.414382 | 0.252059 | 22.45 | True | False | False | False |
| 6 | 0.262511 | 0.311168 | 0.803514 | 0.428356 | 0.314349 | 22.20 | True | False | False | False |
| 7 | 0.230873 | 0.338705 | 0.796495 | 0.412358 | 0.325697 | 23.82 | True | False | False | False |
| 8 | 0.188790 | 0.393009 | 0.789375 | 0.410589 | 0.345532 | 24.06 | True | True | False | False |
| 9 | 0.142537 | 0.453532 | 0.775818 | 0.392860 | 0.372060 | 24.41 | True | True | True | True |
