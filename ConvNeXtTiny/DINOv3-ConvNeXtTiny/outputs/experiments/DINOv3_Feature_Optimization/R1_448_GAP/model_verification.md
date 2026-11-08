# R1_448_GAP model and preprocessing verification

- Input size: 448×448; Test split read: false.
- Original DINOv3 weights SHA-256: `bd30a9459d6149564ef53af6e8a1999980953b009b94cde836ac1bac4d339cb2`.
- Sample Study batch tensor: `[1, 3, 448, 448]`.
- Stage 3: `[1, 384, 28, 28]`; Stage 4: `[1, 768, 14, 14]`.
- Parameters: total 27,827,818, trainable 26,592,778, frozen 1,235,040.

## Trainable module gradient audit

| Module | State | Non-zero gradient |
|---|---|---|
| `backbone.model.downsample_layers.2` | Trainable | True |
| `backbone.model.stages.2` | Trainable | True |
| `backbone.model.downsample_layers.3` | Trainable | True |
| `backbone.model.stages.3` | Trainable | True |
| `backbone.model.norm` | Trainable | True |
| `classifier` | Trainable | True |

- Frozen modules stayed unchanged after one optimizer step: True.
- Representative trainable parameters changed after one optimizer step: True.
- Study BCE from the real Train batch: 0.711556.

## ResizePad checks

| Source W×H | Canvas | Resized W×H | Padding L/T/R/B | Aspect ratio error |
|---:|---:|---:|---:|---:|
| 2544×3056 | 448×448 | 373×448 | [37, 0, 38, 0] | 0.0001286 |
| 3056×2544 | 448×448 | 448×373 | [0, 37, 0, 38] | 0.0001855 |
| 2021×2021 | 448×448 | 448×448 | [0, 0, 0, 0] | 0.0000000 |

Aspect ratio is preserved by scaling the longest side to the target and centering padding on the other axis.
