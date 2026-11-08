# DINOv3 P0 Training Report

- Status: 20/20 epochs; stop reason: `max_epochs_reached`.
- Best epoch: 18; Macro-AUROC: 0.709145; Valid BCE: 0.359553.
- Last epoch: 20; Train BCE: 0.345142; Valid BCE: 0.359767; Macro-AUROC: 0.708963.
- Trainable parameters: 7,690; frozen parameters: 27,820,128.
- Actual mean train throughput: 23.32 images/s.
- Best checkpoint: `best.pth`; last checkpoint: `last.pth`.
- Test split was not read.

| Epoch | Train BCE | Valid BCE | Macro-AUROC | Macro-AUPRC | Macro-F1 | images/s | Train down 3 | Valid up 3 | AUROC no improve 3 | Stop |
|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| 1 | 0.435329 | 0.384747 | 0.640257 | 0.253232 | 0.060150 | 23.82 | False | False | False | False |
| 2 | 0.374860 | 0.384150 | 0.669731 | 0.272585 | 0.087318 | 24.39 | False | False | False | False |
| 3 | 0.366936 | 0.372326 | 0.681954 | 0.281703 | 0.114251 | 24.27 | False | False | False | False |
| 4 | 0.361926 | 0.377085 | 0.687395 | 0.286219 | 0.108710 | 24.24 | True | False | False | False |
| 5 | 0.359169 | 0.366746 | 0.693983 | 0.292760 | 0.098112 | 24.08 | True | False | False | False |
| 6 | 0.356175 | 0.367282 | 0.696641 | 0.296167 | 0.113087 | 24.00 | True | False | False | False |
| 7 | 0.354471 | 0.366127 | 0.701054 | 0.299417 | 0.096853 | 24.02 | True | False | False | False |
| 8 | 0.352830 | 0.365414 | 0.701689 | 0.299436 | 0.125217 | 24.05 | True | False | False | False |
| 9 | 0.351411 | 0.364244 | 0.702760 | 0.300319 | 0.109905 | 23.71 | True | False | False | False |
| 10 | 0.350262 | 0.362676 | 0.704631 | 0.300756 | 0.122443 | 23.43 | True | False | False | False |
| 11 | 0.349543 | 0.362178 | 0.705352 | 0.301653 | 0.126619 | 23.10 | True | False | False | False |
| 12 | 0.348261 | 0.361593 | 0.706413 | 0.302582 | 0.123743 | 23.34 | True | False | False | False |
| 13 | 0.348045 | 0.361772 | 0.706886 | 0.304020 | 0.123595 | 20.77 | True | False | False | False |
| 14 | 0.347042 | 0.360049 | 0.707558 | 0.304898 | 0.128520 | 22.85 | True | False | False | False |
| 15 | 0.346667 | 0.360952 | 0.708301 | 0.305562 | 0.132182 | 22.68 | True | False | False | False |
| 16 | 0.346207 | 0.359868 | 0.709048 | 0.305745 | 0.129888 | 22.66 | True | False | False | False |
| 17 | 0.346013 | 0.360435 | 0.708834 | 0.305978 | 0.122971 | 22.72 | True | False | False | False |
| 18 | 0.345367 | 0.359553 | 0.709145 | 0.306147 | 0.135231 | 22.69 | True | False | False | False |
| 19 | 0.345232 | 0.360245 | 0.708965 | 0.306067 | 0.136133 | 22.78 | True | False | False | False |
| 20 | 0.345142 | 0.359767 | 0.708963 | 0.306201 | 0.124185 | 22.82 | True | False | False | False |
