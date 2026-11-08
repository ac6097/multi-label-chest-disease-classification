# Feature Cache Integrity Report

- Train: 18,513 Studies / 30,344 images / 11,055 Subjects.
- Validation: 859 Studies / 1,422 images / 228 Subjects.
- Subject overlap between Train and Validation: 0.
- Each Study key `(Subject_id, Study_id)` appears once per split and is aligned across R5/R6 and all Stage caches.
- Labels: ten official classes, closed-world encoding, order matches the official class ID order.
- Feature extraction uses each checkpoint in `eval()` and `torch.inference_mode()`; all backbone parameters have `requires_grad=False`.
- No Test CSV, Test image path, or Test label was opened.

## Source Best Checkpoints

| Backbone | Best Epoch | Macro-AUROC in training status | Checkpoint SHA-256 |
|---|---:|---:|---|
| R5 | 6 | 0.803514 | `ee4c09a27b8fb3f39e20c112631a13303bcc8882d669c0bc69f01694a9f4cc7a` |
| R6 | 6 | 0.798802 | `eed4207fbcbfe23ec9357ae274144e3c0cf2b132b181a85a50a7a82210ef3f3f` |

## Cached Study feature dimensions

| Split | Stage 1 | Stage 2 | Stage 3 | Stage 4 | Stage 4 Norm(GAP) |
|---|---:|---:|---:|---:|---:|
| train | 18513×96 | 18513×192 | 18513×384 | 18513×768 | 18513×768 |
| valid | 859×96 | 859×192 | 859×384 | 859×768 | 859×768 |

Stage 1–4 features are GAP vectors computed at each actual Stage output before the next Downsample. Stage 4 Norm(GAP) is the same final normalized pooled vector used by the Global Head. Each Study vector is the arithmetic mean of its per-image vectors. Feature caches include Subject_id, Study_id, Image_Count, labels, source checkpoint SHA-256, stage and extraction metadata.
