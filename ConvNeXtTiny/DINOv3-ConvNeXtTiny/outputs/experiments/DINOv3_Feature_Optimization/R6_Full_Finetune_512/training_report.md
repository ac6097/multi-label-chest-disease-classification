# R6 training report

- Backbone: DINOv3 ConvNeXt-Tiny; input: 512×512; head: GAP + Linear(10).
- Stop: epoch 9/20 (`three_condition_early_stop`).
- Best Epoch: 6; Macro-AUROC: 0.798802.
- Best Train BCE / Valid BCE: 0.265861 / 0.314805.
- Last Train BCE / Valid BCE / Macro-AUROC: 0.147774 / 0.444697 / 0.770623.
- Mean formal train throughput: 22.25 images/s.
- Mean DataLoader wait: 787.0 seconds/epoch.
- Trainable / frozen parameters: 27,827,818 / 0.
- Study-level BCE uses the arithmetic mean of raw image logits within each Study.
- Test split was not read.

| Epoch | Train BCE | Valid BCE | Macro-AUROC | Macro-AUPRC | Macro-F1 | images/s | Train down 3 | Valid up 3 | AUROC no improve 3 | Stop |
|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| 1 | 0.370446 | 0.342752 | 0.732246 | 0.344922 | 0.176354 | 21.73 | False | False | False | False |
| 2 | 0.326163 | 0.328581 | 0.764219 | 0.371181 | 0.200287 | 21.97 | False | False | False | False |
| 3 | 0.313475 | 0.318074 | 0.784108 | 0.401674 | 0.256851 | 22.02 | False | False | False | False |
| 4 | 0.301409 | 0.343876 | 0.779717 | 0.405238 | 0.246401 | 22.20 | True | False | False | False |
| 5 | 0.286723 | 0.310414 | 0.795375 | 0.419290 | 0.241734 | 22.27 | True | False | False | False |
| 6 | 0.265861 | 0.314805 | 0.798802 | 0.432019 | 0.307098 | 22.28 | True | False | False | False |
| 7 | 0.236935 | 0.337166 | 0.796072 | 0.418953 | 0.340707 | 22.79 | True | False | False | False |
| 8 | 0.195866 | 0.386895 | 0.782834 | 0.396789 | 0.325309 | 22.60 | True | True | False | False |
| 9 | 0.147774 | 0.444697 | 0.770623 | 0.380734 | 0.342950 | 22.36 | True | True | True | True |

## R6 trainable scope

R6 is a standalone full-finetuning run. Stem, Stage 1, Stage 2, Stage 3, Stage 4, final normalization, and Global Head are trainable; no Local Head is present. All trainable parameters are included in AdamW.
