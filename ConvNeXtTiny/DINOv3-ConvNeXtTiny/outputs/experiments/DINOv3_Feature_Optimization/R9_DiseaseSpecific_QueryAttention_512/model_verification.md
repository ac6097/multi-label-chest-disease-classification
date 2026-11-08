# R9_DiseaseSpecific_QueryAttention_512 model and preprocessing verification

- Input size: 512×512; Test split read: false.
- Original DINOv3 weights SHA-256: `bd30a9459d6149564ef53af6e8a1999980953b009b94cde836ac1bac4d339cb2`.
- Sample Study batch tensor: `[1, 3, 512, 512]`.
- Stage 3: `[1, 384, 32, 32]`; Stage 4: `[1, 768, 16, 16]`.
- Parameters: total 27,929,726, trainable 26,694,686, frozen 1,235,040.

## Trainable module gradient audit

| Module | State | Non-zero gradient |
|---|---|---|
| `backbone.model.downsample_layers.2` | Trainable | True |
| `backbone.model.stages.2` | Trainable | True |
| `backbone.model.downsample_layers.3` | Trainable | True |
| `backbone.model.stages.3` | Trainable | True |
| `backbone.model.norm` | Trainable | True |
| `classifier` | Trainable | True |
| `query_attention` | Trainable | True |

- Frozen modules stayed unchanged after one optimizer step: True.
- Representative trainable parameters changed after one optimizer step: True.
- Study BCE from the real Train batch: 0.705066.

## ResizePad checks

| Source W×H | Canvas | Resized W×H | Padding L/T/R/B | Aspect ratio error |
|---:|---:|---:|---:|---:|
| 2544×3056 | 512×512 | 426×512 | [43, 0, 43, 0] | 0.0004295 |
| 3056×2544 | 512×512 | 512×426 | [0, 43, 0, 43] | 0.0006201 |
| 2021×2021 | 512×512 | 512×512 | [0, 0, 0, 0] | 0.0000000 |

## Group-specific checks

- stage3_shape: `[1, 384, 32, 32]`
- stage4_shape: `[1, 768, 16, 16]`
- alpha_init_max_abs_error: `0.0`
- initial_zero_branch_logit_max_abs_error: `0.0`
- attended_shape: `[1, 10, 128]`

Aspect ratio is preserved by scaling the longest side to the target and centering padding on the other axis.
