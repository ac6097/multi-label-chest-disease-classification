# P0/P1/P2 Preflight Audit

## Data and output contract

- Train: 18,513 Studies / 30,344 images; Valid: 859 Studies / 1,422 images.
- Train/Valid Study-key overlap: 0.
- Label order is IDs 1–10 matching `LABEL_NAMES`: `Enlarged Cardiomediastinum, Pneumothorax, Consolidation, Pneumonia, Edema, Cardiomegaly, Atelectasis, Lung Opacity, Pleural Effusion, No Finding`.
- Dataset encodes each positive class ID at index `class_id - 1`; closed-world known masks come from the official configuration.
- Inputs use 320×320 ResizePad, grayscale-to-three-channel conversion, the existing light training augmentation, and DINOv3 ImageNet normalization.
- Only official Train and Valid manifests are loaded. Test is not read by preflight or training.

## Historical D0 loss-path audit

- Train and validation use the same `study_logit_mean` and `study_bce_per_study` functions. Both compute BCE over per-Study mean image logits and average over known labels within each Study.
- Epoch Train BCE is the sum of per-Study BCE divided by eligible Study count; validation loss uses the same per-Study reduction and denominator.
- Gradient accumulation divides each microbatch Study-loss sum by the total eligible Study count in the complete accumulation window. No second division by accumulation steps is applied.
- The Study sampler preserves every image of each Study and asserts each Study is visited exactly once per epoch. D0 logs match the dataset counts for all 20 completed epochs.
- D0 Train BCE moved from 0.371992 at Epoch 1 to 0.001015 at Epoch 20; Valid BCE moved from 0.348269 to 1.117882; final Macro-AUROC was 0.744670 (best Epoch 5: 0.795868).
- The audited implementation shows no label-order mismatch, Study/image reduction mismatch, accumulation double-scaling, or Study repetition. D0's near-zero late Train BCE with rising Valid BCE is consistent with observed overfitting; no implementation defect was found in these paths.

## Historical D0 evidence

- D0 complete: True; logged epochs: 20; per-epoch counts match 18,513 Studies / 30,344 images: True.
- D0 is read as a historical reference only; it is not retrained or modified.
- Official DINOv3 initialization SHA-256: `bd30a9459d6149564ef53af6e8a1999980953b009b94cde836ac1bac4d339cb2`; missing/unexpected tensors: 0/0.

## Environment

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU (8.00 GiB). CUDA 12.8; BF16 supported: True.
- PyTorch 2.10.0+cu128; Python 3.12.3; CPU logical processors: 32.
