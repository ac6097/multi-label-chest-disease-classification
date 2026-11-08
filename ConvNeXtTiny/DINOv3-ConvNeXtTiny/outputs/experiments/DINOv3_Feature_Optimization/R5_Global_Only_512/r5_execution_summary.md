# R5 Global-only execution summary

- Status: complete; stop reason: `three_condition_early_stop` at epoch 9/20.
- Best epoch: 6; Macro-AUROC: 0.803514.
- Runtime copied from R3: budget=8, accumulation=4, channels-last=True, checkpointing=False; no runtime benchmark search.
- Actual-runtime disposable backward preflight passed on 31 images / 17 Studies across 4 accumulation microbatches; no optimizer step was counted in preflight.
- R3 through Best Epoch had 6037 formal optimizer updates.
- R5 trainable/frozen parameters: 26,592,778 / 1,235,040.
- Standalone inference package: `\\192.168.60.1\团队文件-Public\multi-label-chest-disease-classification\ConvNeXtTiny\DINOv3-ConvNeXtTiny\outputs\experiments\DINOv3_Feature_Optimization\R5_Global_Only_512\submission_export\R5_Inference.zip` (103,620,163 bytes; strict size check passed).
- Full Validation inference replay passed: Macro-AUROC 0.803514; max probability error 1.1920929e-07.
- Only official Train and Validation were used. Test was not read by training or model selection.

