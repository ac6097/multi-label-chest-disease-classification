# R9 training report

- Backbone: DINOv3 ConvNeXt-Tiny DINOv3ConvNeXtTinyR9DiseaseSpecificQueryAttention; input: 512×512; head: GAP + Linear(10).
- Stop: epoch 9/20 (`three_condition_early_stop`).
- Best Epoch: 6; Macro-AUROC: 0.802962.
- Best Train BCE / Valid BCE: 0.262541 / 0.311438.
- Last Train BCE / Valid BCE / Macro-AUROC: 0.142414 / 0.454692 / 0.775063.
- Mean formal train throughput: 22.21 images/s.
- Mean DataLoader wait: 1022.9 seconds/epoch.
- Trainable / frozen parameters: 26,694,686 / 1,235,040.
- Study-level BCE uses the arithmetic mean of raw image logits within each Study.
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

## R9 auxiliary branch

10 learnable disease queries cross-attend 32x32 Stage3 tokens (d_model=128, 4 heads); per-disease zero-initialized logit head; z_k = z4_k + sigmoid(theta_k) * z_attn_k

The original R5 Global Head and main path are unchanged; the added state namespaces are `query_attention, fusion_alpha`. Per-epoch branch diagnostics are in `branch_diagnostics.csv`, `alpha_by_epoch.csv`, `stage3_logit_contribution.csv`, `attention_statistics.csv`, `query_similarity.csv`.
