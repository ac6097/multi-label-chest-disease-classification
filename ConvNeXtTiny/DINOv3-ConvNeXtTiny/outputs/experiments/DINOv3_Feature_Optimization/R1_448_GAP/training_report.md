# DINOv3 P2 Training Report

- Status: 9/20 epochs; stop reason: `three_condition_early_stop`.
- Best epoch: 6; Macro-AUROC: 0.799318; Valid BCE: 0.318269.
- Last epoch: 9; Train BCE: 0.151519; Valid BCE: 0.459910; Macro-AUROC: 0.766058.
- Trainable parameters: 26,592,778; frozen parameters: 1,235,040.
- Actual mean train throughput: 22.32 images/s.
- Best checkpoint: `best.pth`; last checkpoint: `last.pth`.
- Test split was not read.

| Epoch | Train BCE | Valid BCE | Macro-AUROC | Macro-AUPRC | Macro-F1 | images/s | Train down 3 | Valid up 3 | AUROC no improve 3 | Stop |
|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| 1 | 0.373346 | 0.359830 | 0.729257 | 0.339706 | 0.175543 | 21.54 | False | False | False | False |
| 2 | 0.327592 | 0.331335 | 0.764427 | 0.370112 | 0.221185 | 18.61 | False | False | False | False |
| 3 | 0.312434 | 0.321475 | 0.783406 | 0.398723 | 0.248432 | 23.14 | False | False | False | False |
| 4 | 0.299318 | 0.345032 | 0.781636 | 0.408703 | 0.221457 | 23.70 | True | False | False | False |
| 5 | 0.284525 | 0.313692 | 0.792579 | 0.413470 | 0.252703 | 22.82 | True | False | False | False |
| 6 | 0.263042 | 0.318269 | 0.799318 | 0.417669 | 0.318065 | 21.33 | True | False | False | False |
| 7 | 0.233160 | 0.338387 | 0.788106 | 0.406759 | 0.308207 | 23.32 | True | False | False | False |
| 8 | 0.194581 | 0.386828 | 0.776866 | 0.386844 | 0.332673 | 23.27 | True | True | False | False |
| 9 | 0.151519 | 0.459910 | 0.766058 | 0.374821 | 0.355237 | 23.11 | True | True | True | True |
