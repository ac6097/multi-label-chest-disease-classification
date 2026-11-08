# R10 training report

- Backbone: DINOv3 ConvNeXt-Tiny DINOv3ConvNeXtTinyR10Stage4GuidedCrossStageAttention; input: 512×512; head: GAP + Linear(10).
- Stop: epoch 9/20 (`three_condition_early_stop`).
- Best Epoch: 6; Macro-AUROC: 0.803098.
- Best Train BCE / Valid BCE: 0.264361 / 0.309637.
- Last Train BCE / Valid BCE / Macro-AUROC: 0.146686 / 0.445177 / 0.772915.
- Mean formal train throughput: 22.59 images/s.
- Mean DataLoader wait: 997.8 seconds/epoch.
- Trainable / frozen parameters: 26,794,654 / 1,235,040.
- Study-level BCE uses the arithmetic mean of raw image logits within each Study.
- Test split was not read.

| Epoch | Train BCE | Valid BCE | Macro-AUROC | Macro-AUPRC | Macro-F1 | images/s | Train down 3 | Valid up 3 | AUROC no improve 3 | Stop |
|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| 1 | 0.371808 | 0.343314 | 0.733904 | 0.343269 | 0.160357 | 22.96 | False | False | False | False |
| 2 | 0.327353 | 0.330448 | 0.769653 | 0.376115 | 0.209820 | 21.44 | False | False | False | False |
| 3 | 0.312203 | 0.319677 | 0.786724 | 0.399684 | 0.279624 | 21.51 | False | False | False | False |
| 4 | 0.299373 | 0.344738 | 0.787471 | 0.409661 | 0.236020 | 22.29 | True | False | False | False |
| 5 | 0.285229 | 0.309026 | 0.798639 | 0.416125 | 0.252638 | 22.86 | True | False | False | False |
| 6 | 0.264361 | 0.309637 | 0.803098 | 0.435072 | 0.317655 | 22.52 | True | False | False | False |
| 7 | 0.234417 | 0.332935 | 0.799576 | 0.414948 | 0.320007 | 22.74 | True | False | False | False |
| 8 | 0.193195 | 0.391811 | 0.789945 | 0.406206 | 0.346217 | 23.01 | True | True | False | False |
| 9 | 0.146686 | 0.445177 | 0.772915 | 0.387667 | 0.365015 | 23.97 | True | True | True | True |

## R10 auxiliary branch

R9 plus q_k(x) = e_k + W_q LayerNorm(g4(x)) with zero-initialized W_q: Stage4 global semantics condition Stage3 token selection

The original R5 Global Head and main path are unchanged; the added state namespaces are `query_attention, fusion_alpha`. Per-epoch branch diagnostics are in `branch_diagnostics.csv`, `alpha_by_epoch.csv`, `stage3_logit_contribution.csv`, `attention_statistics.csv`, `query_similarity.csv`, `stage4_query_conditioning.csv`.
