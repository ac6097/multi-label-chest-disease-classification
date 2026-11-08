# DINOv3 feature optimization final comparison

| Group | Backbone | Resolution | Head | Best Macro-AUROC | Mean train images/s | Stop Epoch | Stop reason | Status |
|---|---|---:|---|---:|---:|---:|---|---|
| R0 | ConvNeXt-Tiny | 320 | GAP | 0.794882 | 23.58 | 8 | P2 baseline | historical baseline |
| R1 | ConvNeXt-Tiny | 448 | GAP | 0.799318 | 22.32 | 9 | three_condition_early_stop | complete |
| R2 | ConvNeXt-Tiny | 512 | GAP | 0.802116 | 22.99 | 9 | three_condition_early_stop | complete |
| R3 | ConvNeXt-Tiny | 512 | GAP+Local | 0.804653 | 21.45 | 9 | three_condition_early_stop | complete |
| R4 | ConvNeXt-Small | 512 | GAP | 0.799287 | 24.17 | 9 | three_condition_early_stop | complete |

## Statistical comparisons

Paired Subject-cluster confidence intervals are in `paired_subject_bootstrap.json`; comparisons whose interval includes zero are not described as clear improvements.

## Evaluation scope

All reported model selection and analysis uses the official Train/Validation split. Test split was not read.
