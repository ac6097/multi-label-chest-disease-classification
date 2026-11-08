# Hybrid H0/H1/H2 comparison

## Best-checkpoint summary

| Model | Additional self-supervision | Best Epoch | Macro-AUROC | Pneumonia AUROC | Lung Opacity AUROC | Parameters | Train images/s at best epoch |
|---|---|---:|---:|---:|---:|---:|---:|
| H0 | None | 3 | 0.779060 | 0.713830 | 0.709462 | 49,620,490 | 24.94 |
| H1 | Pixel reconstruction | 6 | 0.793342 | 0.722252 | 0.712168 | 49,620,490 | 24.26 |
| H2 | Latent feature reconstruction | 9 | 0.773039 | 0.682048 | 0.701626 | 49,620,490 | 24.32 |

## Epoch 7 to Epoch 20 generalization trend

| Model | AUROC Epoch 7 | AUROC Epoch 20 | AUROC change 7→20 | Train-Val gap Epoch 7 | Train-Val gap Epoch 20 |
|---|---:|---:|---:|---:|---:|
| H0 | 0.772427 | 0.738428 | -0.033999 | +0.081915 | +0.448234 |
| H1 | 0.780480 | 0.758202 | -0.022279 | +0.044935 | +0.318461 |
| H2 | 0.767322 | 0.753982 | -0.013340 | +0.019167 | +0.155899 |

Highest observed validation Macro-AUROC: **H1** at epoch 6 (0.793342).
All three classifiers completed the configured 20 epochs; selection used Validation Study Macro-AUROC.

## Paired Subject bootstrap

Resampled 228 Subjects with replacement over 859 Validation Studies; seed=42, requested replicates=2000. Invalid replicates rejected for a missing class polarity: 0.
| Difference | Δ Macro-AUROC | Paired 95% CI | Valid replicates |
|---|---:|---:|---:|
| H1-H0 | +0.014282 | [-0.001115, +0.026949] | 2000 |
| H2-H0 | -0.006021 | [-0.022695, +0.009142] | 2000 |
| H2-H1 | -0.020303 | [-0.033467, -0.005284] | 2000 |

## Interpretation

- H1−H0 isolates the added official-train pixel reconstruction stage under a shared downstream classifier architecture; observed Δ AUROC is +0.014282.
- H2−H0 compares the latent Stage3/Block6 reconstruction stage with no additional pretraining; observed Δ AUROC is -0.006021.
- H2−H1 compares the two self-supervised targets; observed Δ AUROC is -0.020303.
- Pneumonia AUROC: H0=0.713830, H1=0.722252, H2=0.682048.
- Lung Opacity AUROC: H0=0.709462, H1=0.712168, H2=0.701626.
- Compare epoch 5–7 peaks and subsequent declines in the curves. The absolute train/validation losses use the same BCE but train augmentation differs from validation.
- Historical ConvNeXt-Tiny scores 0.791286 and 0.793451 are context only; those runs used different architecture/training conditions and are not matched ablation baselines.
- H0/H1/H2 share the same ImageNet initial checkpoint, classifier architecture, downstream split, loss and selected supervised budget. H1/H2 each used the same number of official Train images and epochs; self-supervised optimizer updates can differ because their measured microbatch configurations differ.
- No Test CSV, Test image, or submission metric was read or generated.

## Artifacts

- `hybrid_gpu_benchmark.md` and `.csv`: real-data training throughput selection.
- `hybrid_h0_report.md`, `hybrid_h1_report.md`, `hybrid_h2_report.md`: per-run training and pretraining details.
- `hybrid_macro_auroc.png`, `hybrid_validation_loss.png`, `hybrid_train_val_gap.png`, `hybrid_best_per_class_auroc.png`, `hybrid_images_per_second.png`, `hybrid_peak_memory.png`.
- `paired_subject_bootstrap.json`: paired confidence intervals.
