# R7 Architecture and Initialization Report

- Backbone: official general-vision DINOv3 ConvNeXt-Tiny; same pinned revision and SHA-256 as R5.
- Original pretrained backbone SHA-256: `bd30a9459d6149564ef53af6e8a1999980953b009b94cde836ac1bac4d339cb2`.
- Global Head independently initialized with seed 42; initial hash matches R5: `40dc9552892097293310d9fca16e61913c5fa11eae96e3d56e572553141827a9`.
- Original R5 Global Head and backbone parameter names are unchanged; the added state namespace is `transfer.*`.
- Main path: Stage3 → original Downsample → Stage4.
- Transfer path: Stage3 `[B,384,32,32]` → AvgPool2d(2,2) → Conv2d(384,768,1,bias=False) → gamma scaling.
- Fusion: `F4_fused = F4 + gamma*T(F3)` before the original Stage4 pooled-token normalization and Global Linear(768,10) head.
- No added activation, attention, local head, auxiliary loss, multiscale module, or pretraining phase.
- Gamma initialization equivalence CPU FP32: `{'single_image_logits_max_abs_error': 0.0, 'multi_image_logits_max_abs_error': 0.0, 'study_logit_mean_max_abs_error': 0.0, 'study_probability_max_abs_error': 0.0, 'stage3_shape': [6, 384, 32, 32], 'stage4_shape': [6, 768, 16, 16], 'projection_shape': [6, 768, 16, 16], 'gamma_init': 0.0}`.
- Gamma initialization equivalence CUDA BF16: `{'single_image_logits_max_abs_error': 0.0, 'multi_image_logits_max_abs_error': 0.0, 'study_logit_mean_max_abs_error': 0.0, 'study_probability_max_abs_error': 0.0, 'amp_dtype': 'bfloat16'}`.
- Added parameters: Projection 294,912; gamma 1.
- Total/trainable/frozen parameters: 28,122,731/26,887,691/1,235,040.
- R5 selected runtime reused exactly: image budget 8, accumulation 4, channels-last True, checkpointing False; no new benchmark or LR search.
- Study Logit Mean, Study BCE, ResizePad, augmentation, normalization, split and three-condition stop rule follow the R5 implementation.
- Actual two-step runtime preflight passed at 1.50 GiB peak reserved memory; initial gamma and delayed projection gradients were verified.
- Only official Train/Validation were accessed; Test was not read.
