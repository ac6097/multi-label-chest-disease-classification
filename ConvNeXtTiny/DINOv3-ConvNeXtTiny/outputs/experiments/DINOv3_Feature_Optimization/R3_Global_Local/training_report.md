# R3 training report

- Backbone: DINOv3 ConvNeXt-Tiny; input: 512×512; head: GAP + mask-aware Stage3 Top-K.
- Stop: epoch 9/20 (`three_condition_early_stop`).
- Best Epoch: 6; Macro-AUROC: 0.804653.
- Best Train BCE / Valid BCE: 0.263346 / 0.312023.
- Last Train BCE / Valid BCE / Macro-AUROC: 0.142984 / 0.440321 / 0.768128.
- Mean formal train throughput: 21.45 images/s.
- Mean DataLoader wait: 1048.6 seconds/epoch.
- Trainable / frozen parameters: 26,596,629 / 1,235,040.
- Study-level BCE uses the arithmetic mean of raw image logits within each Study.
- Test split was not read.

| Epoch | Train BCE | Valid BCE | Macro-AUROC | Macro-AUPRC | Macro-F1 | images/s | Train down 3 | Valid up 3 | AUROC no improve 3 | Stop |
|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| 1 | 0.371995 | 0.346927 | 0.730814 | 0.341079 | 0.182874 | 20.53 | False | False | False | False |
| 2 | 0.325350 | 0.327056 | 0.768889 | 0.376526 | 0.201673 | 22.43 | False | False | False | False |
| 3 | 0.311120 | 0.321315 | 0.787287 | 0.398871 | 0.275050 | 22.45 | False | False | False | False |
| 4 | 0.299633 | 0.341910 | 0.787990 | 0.410656 | 0.243721 | 22.42 | True | False | False | False |
| 5 | 0.284890 | 0.309799 | 0.799099 | 0.421186 | 0.251869 | 22.40 | True | False | False | False |
| 6 | 0.263346 | 0.312023 | 0.804653 | 0.418717 | 0.310102 | 22.25 | True | False | False | False |
| 7 | 0.231466 | 0.334954 | 0.801599 | 0.410623 | 0.311081 | 22.33 | True | False | False | False |
| 8 | 0.189766 | 0.392132 | 0.785323 | 0.393479 | 0.326124 | 22.49 | True | True | False | False |
| 9 | 0.142984 | 0.440321 | 0.768128 | 0.380528 | 0.361256 | 15.78 | True | True | True | True |
