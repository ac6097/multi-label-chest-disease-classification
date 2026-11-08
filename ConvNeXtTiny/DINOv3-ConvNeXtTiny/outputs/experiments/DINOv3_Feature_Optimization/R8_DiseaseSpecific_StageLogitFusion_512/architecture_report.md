# R8 Architecture and Initialization Report

- Backbone: official general-vision DINOv3 ConvNeXt-Tiny; same pinned revision and SHA-256 as R5.
- Original pretrained backbone SHA-256: `bd30a9459d6149564ef53af6e8a1999980953b009b94cde836ac1bac4d339cb2`.
- Original Global Head independently initialized with seed 42; initial hash matches R5: `40dc9552892097293310d9fca16e61913c5fa11eae96e3d56e572553141827a9`.
- Original R5 parameter names and main path are unchanged; the added state namespaces are `stage3_branch, fusion_alpha`.
- Main path: Stage3 -> original Downsample -> Stage4 -> official pooled-token normalization -> Linear(768,10) -> z4.
- Added branch: LayerNorm(384) + Linear(384,10) on Stage3 GAP, zero-initialized head, z_k = z4_k + sigmoid(theta_k) * z3_k.
- Head: original Global Head + zero-init Stage3 GAP auxiliary head.
- No R7 shared residual transfer, no auxiliary loss, no multiscale module, no extra Stage3 structure, no Study-level pooling change.
- New parameters: 4,628; total/trainable/frozen: 27,832,446/26,597,406/1,235,040.
- R5 selected runtime reused exactly: image budget 8, accumulation 4, channels-last True, gradient checkpointing False; no new benchmark or LR search.
- Runtime note: because Studies are indivisible, the measured effective batch is 30.5 images / 18.5 Studies per optimizer update instead of exactly 32 images. The LR schedule (2 warmup epochs + cosine over the remaining 18 planned epochs) is anchored to the 20-epoch update budget, so an early stop leaves LR above its configured minimum; this is identical to R5/R7 and is not a R8/R9/R10-specific change.
- Study Logit Mean, Study BCE, ResizePad, augmentation, normalization, split, seed and the three-condition stop rule follow the R5 implementation.
- Initialization equivalence: CPU FP32 max |logit| error 0.000e+00, CUDA BF16 0.000e+00.
- Delayed-gradient contract verified over 2 real optimizer steps; preflight peak reserved 1.42 GiB.
- Only official Train/Validation were accessed; Test was not read.

## Delayed gradient checks

| Check | Passed |
|---|---|
| stage3_auxiliary_path_reaches_stage3_trunk | True |
| stage3_auxiliary_path_leaves_main_classifier_gradient_zero | True |
| stage3_branch_head_gradient_at_step1 | True |
| theta_gradient_at_step1_is_zero | True |
| stage3_branch_norm_gradient_at_step1_is_zero | True |
| theta_gradient_at_step2_is_nonzero | True |
| stage3_branch_norm_gradient_at_step2_is_nonzero | True |
| stage4_main_path_gradient_at_step1 | True |
