# Hybrid H2 report

## Self-supervised stage

# Hybrid H2_LATENT self-supervised pretraining

- Status: completed 20 epochs
- Epochs completed: 20 / 20
- Best epoch by training reconstruction loss: 20
- Best total / CNN / ViT reconstruction loss: 0.015656 / 0.000018 / 0.031294
- Encoder export for supervised fine-tuning: `\\192.168.60.1\团队文件-Public\multi-label-chest-disease-classification\ConvNeXtTiny\outputs\hybrid\h2_pretrain\encoder_best.pth`
- Best/Last full checkpoint bytes: 618,844,664/618,844,664; best/last encoder export bytes: 198,577,923/198,577,923
- Initial ImageNet checkpoint SHA-256: `d11bb1f885df2a43b8160531e593bd76cc1bbc9bb724ca51e599afcf1073e121`
- Train image manifest SHA-256: `17077eb7d204f71eaf845ba7f88050437e75a21ac01a2b9dc4f79236ee5afb2f`
- Pretraining parameters: CNN 27,864,960; ViT 21,744,000; CNN/ViT decoders 1,622,784/296,448; total 51,528,960.
- Data source: official train images only; disease label columns were not read; valid/test were not used.
- Target effective batch: 32 images; each epoch traverses every official train image exactly once.
- Decoder state is retained in resumable best/last pretraining checkpoints and excluded from encoder export.
- Encoder/decoder gradient and parameter-change diagnostics are recorded on the first optimizer update of each epoch; non-finite/zero gradients or collapsed features/outputs stop the run with a failure checkpoint.
- Mask: exactly 60 of 100 random 32×32 regions; masks are keyed by Subject/Study/image ID so H1/H2 share positions even when microbatch sizes differ; each maps to 2×2 positions on the Stage3 and ViT 20×20 grids.
- CNN target is a detached Stage3 feature map; masked Stage3 output is replaced before Stage4, then reconstructed.
- ViT target is a detached output of Block 6; CLS is excluded from masking and reconstruction; masked patch tokens continue through Blocks 7–12.
- Stage3 and ViT Block 1–6 features are computed before the mask and may contain contextual information from all input regions. This is intermediate-feature reconstruction, not input-level MAE.
- No Teacher–Student/EMA, distillation, or contrastive objective was used.
- Diagnostics record branch/segment gradients, parameter changes, feature spatial variance, and decoder output variance.
- Visuals are latent feature/error maps, not reconstructed X-rays.

## Epoch log

| Epoch | Total reconstruction loss | CNN loss | ViT loss | Mask ratio | Images/s | Peak allocated/reserved GiB | Encoder gradient norms |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.459425 | 0.160683 | 0.758166 | 0.600 | 21.96 | 1.378/1.586 | S3=0.01519, S4=1.265, ViT pre/post=0.2184/1.983, decoders=5.364/1.009 |
| 2 | 0.172574 | 0.016226 | 0.328922 | 0.600 | 22.44 | 1.378/1.586 | S3=0.0001543, S4=0.01465, ViT pre/post=0.1197/0.1725, decoders=0.05327/0.07571 |
| 3 | 0.113679 | 0.009474 | 0.217883 | 0.600 | 23.04 | 1.378/1.586 | S3=0.007286, S4=0.009957, ViT pre/post=0.1174/0.1347, decoders=0.06977/0.06232 |
| 4 | 0.086346 | 0.000301 | 0.172391 | 0.600 | 22.42 | 1.378/1.586 | S3=8.328e-06, S4=0.0001259, ViT pre/post=0.1232/0.1125, decoders=0.001338/0.05232 |
| 5 | 0.073603 | 0.000298 | 0.146908 | 0.600 | 23.09 | 1.378/1.586 | S3=2.584e-05, S4=0.0004767, ViT pre/post=0.1016/0.1479, decoders=0.003843/0.04381 |
| 6 | 0.063681 | 0.001803 | 0.125559 | 0.600 | 23.04 | 1.378/1.586 | S3=1.373e-05, S4=0.0004164, ViT pre/post=0.1215/0.1551, decoders=0.002619/0.03828 |
| 7 | 0.054878 | 0.000284 | 0.109472 | 0.600 | 23.17 | 1.378/1.586 | S3=1.532e-05, S4=0.0002616, ViT pre/post=0.9751/0.2869, decoders=0.001571/0.03822 |
| 8 | 0.042559 | 0.000393 | 0.084726 | 0.600 | 23.06 | 1.378/1.586 | S3=2.215e-05, S4=0.0003784, ViT pre/post=0.1372/0.1604, decoders=0.003556/0.03176 |
| 9 | 0.043262 | 0.021533 | 0.064990 | 0.600 | 22.41 | 1.378/1.586 | S3=5.098e-05, S4=0.0004843, ViT pre/post=0.1257/0.2371, decoders=0.004349/0.02919 |
| 10 | 0.027585 | 0.000030 | 0.055140 | 0.600 | 23.17 | 1.378/1.586 | S3=7.816e-06, S4=6.035e-05, ViT pre/post=0.1052/0.1361, decoders=0.0004979/0.03013 |
| 11 | 0.023929 | 0.000023 | 0.047834 | 0.600 | 22.38 | 1.378/1.586 | S3=6.611e-06, S4=0.000101, ViT pre/post=0.08693/0.09495, decoders=0.0001771/0.02261 |
| 12 | 0.020932 | 0.000020 | 0.041843 | 0.600 | 23.07 | 1.378/1.586 | S3=3.851e-06, S4=8.041e-05, ViT pre/post=0.1062/0.1296, decoders=0.0001792/0.02097 |
| 13 | 0.019571 | 0.000017 | 0.039124 | 0.600 | 22.50 | 1.378/1.586 | S3=3.265e-06, S4=7.553e-05, ViT pre/post=0.07727/0.09216, decoders=0.0003933/0.02071 |
| 14 | 0.018238 | 0.000017 | 0.036459 | 0.600 | 23.09 | 1.378/1.586 | S3=2.233e-06, S4=5.114e-05, ViT pre/post=0.09057/0.1477, decoders=0.0003356/0.02385 |
| 15 | 0.017533 | 0.000017 | 0.035049 | 0.600 | 22.66 | 1.378/1.586 | S3=2.811e-06, S4=5.087e-05, ViT pre/post=0.05094/0.07281, decoders=0.0003134/0.01551 |
| 16 | 0.016994 | 0.000017 | 0.033972 | 0.600 | 21.73 | 1.378/1.586 | S3=4.118e-06, S4=8.717e-05, ViT pre/post=0.05251/0.07286, decoders=0.0002987/0.01672 |
| 17 | 0.016554 | 0.000016 | 0.033091 | 0.600 | 21.91 | 1.378/1.586 | S3=4.796e-06, S4=0.0001024, ViT pre/post=0.06086/0.07476, decoders=0.0005324/0.01874 |
| 18 | 0.016148 | 0.000016 | 0.032280 | 0.600 | 22.01 | 1.378/1.586 | S3=2.862e-06, S4=4.511e-05, ViT pre/post=0.04479/0.07519, decoders=0.000265/0.01623 |
| 19 | 0.015902 | 0.000015 | 0.031789 | 0.600 | 17.83 | 1.378/1.586 | S3=1.813e-06, S4=3.47e-05, ViT pre/post=0.05236/0.05286, decoders=0.0002194/0.0148 |
| 20 | 0.015656 | 0.000018 | 0.031294 | 0.600 | 22.03 | 1.378/1.586 | S3=1.753e-06, S4=5.373e-05, ViT pre/post=0.04429/0.04906, decoders=0.0003032/0.01454 |


## Classification fine-tuning

# Hybrid H2 classification report

- Status: completed 20 epochs
- Best Epoch: 9
- Best Validation Macro-AUROC: 0.773039
- Best Train Loss / Valid Loss: 0.295760 / 0.325913
- Best Macro-AUPRC / Macro-F1: 0.384059 / 0.255692
- Best checkpoint: `\\192.168.60.1\团队文件-Public\multi-label-chest-disease-classification\ConvNeXtTiny\outputs\hybrid\h2\best.pth`
- Last checkpoint: `\\192.168.60.1\团队文件-Public\multi-label-chest-disease-classification\ConvNeXtTiny\outputs\hybrid\h2\last.pth`
- Test split was not read.

## Configuration and provenance

- ConvNeXtV2: `convnextv2_tiny.fcmae_ft_in1k`; ViT: `vit_small_patch16_224.augreg_in1k`
- ImageNet initialization SHA-256: `d11bb1f885df2a43b8160531e593bd76cc1bbc9bb724ca51e599afcf1073e121`
- Train/Validation manifest SHA-256: `17077eb7d204f71eaf845ba7f88050437e75a21ac01a2b9dc4f79236ee5afb2f` / `d5dbd8b83413a2c029684f8e374808dc2b1c51f6e859f4c594c11043f8a9e1fd`
- Device: `NVIDIA GeForce RTX 4060 Laptop GPU`; PyTorch `2.10.0+cu128`; timm `1.0.30`
- Parameters: CNN 27,864,960; ViT 21,744,000; head 11,530; total 49,620,490
- Study outputs are the arithmetic mean of image-level raw logits; the loss is unweighted Standard BCE.
- Input is 320×320 ResizePad and one shared light augmentation, then each branch uses its own ImageNet normalization.

## Per-epoch classification metrics

- Classifier FLOPs per image (torch.profiler estimate): `36923011986`
- Best / Last checkpoint size: 595,922,655 / 595,922,655 bytes
| Epoch | Train Loss | Valid Loss | Macro-AUROC | Macro-AUPRC | Macro-F1 | CNN/ViT/Head LR | Train images/s | Val Studies/s | Val Images/s | Peak Allocated/Reserved GiB |
|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
| 1 | 0.550998 | 0.373897 | 0.681693 | 0.286550 | 0.077809 | 4.968e-09/4.968e-09/9.935e-08 | 22.54 | 9.17 | 15.18 | 3.074/3.170 |
| 2 | 0.356931 | 0.360082 | 0.716286 | 0.311129 | 0.087051 | 4.998e-06/4.998e-06/9.995e-05 | 23.54 | 17.12 | 28.35 | 3.074/3.205 |
| 3 | 0.341591 | 0.343737 | 0.740808 | 0.333364 | 0.169039 | 1.000e-05/1.000e-05/2.000e-04 | 24.81 | 18.81 | 31.14 | 3.074/3.227 |
| 4 | 0.330952 | 0.347371 | 0.739397 | 0.343975 | 0.185114 | 9.928e-06/9.928e-06/1.986e-04 | 24.60 | 18.47 | 30.58 | 3.074/3.227 |
| 5 | 0.323066 | 0.330749 | 0.758490 | 0.368396 | 0.182800 | 9.713e-06/9.713e-06/1.943e-04 | 24.86 | 18.81 | 31.14 | 3.074/3.227 |
| 6 | 0.315247 | 0.328431 | 0.767878 | 0.378003 | 0.218554 | 9.363e-06/9.363e-06/1.873e-04 | 24.83 | 17.98 | 29.76 | 3.074/3.227 |
| 7 | 0.309674 | 0.328841 | 0.767322 | 0.370764 | 0.203799 | 8.889e-06/8.889e-06/1.778e-04 | 21.08 | 18.45 | 30.54 | 3.074/3.227 |
| 8 | 0.302862 | 0.331966 | 0.764171 | 0.371675 | 0.247054 | 8.303e-06/8.303e-06/1.661e-04 | 24.77 | 18.96 | 31.39 | 3.074/3.227 |
| 9 | 0.295760 | 0.325913 | 0.773039 | 0.384059 | 0.255692 | 7.624e-06/7.624e-06/1.525e-04 | 24.32 | 18.19 | 30.11 | 3.074/3.227 |
| 10 | 0.288190 | 0.329885 | 0.770552 | 0.369102 | 0.234260 | 6.874e-06/6.874e-06/1.375e-04 | 24.82 | 18.38 | 30.43 | 3.074/3.227 |
| 11 | 0.279233 | 0.329444 | 0.769635 | 0.381727 | 0.275768 | 6.075e-06/6.075e-06/1.215e-04 | 24.44 | 18.48 | 30.59 | 3.074/3.227 |
| 12 | 0.269636 | 0.336176 | 0.765966 | 0.375562 | 0.278720 | 5.250e-06/5.250e-06/1.050e-04 | 24.86 | 18.34 | 30.36 | 3.074/3.227 |
| 13 | 0.259566 | 0.338340 | 0.762843 | 0.378058 | 0.287684 | 4.426e-06/4.426e-06/8.851e-05 | 24.65 | 18.32 | 30.33 | 3.074/3.227 |
| 14 | 0.249593 | 0.346490 | 0.760195 | 0.375468 | 0.279040 | 3.626e-06/3.626e-06/7.252e-05 | 25.05 | 18.50 | 30.63 | 3.074/3.227 |
| 15 | 0.241070 | 0.350632 | 0.760419 | 0.374046 | 0.306110 | 2.874e-06/2.874e-06/5.748e-05 | 25.11 | 18.78 | 31.08 | 3.074/3.227 |
| 16 | 0.233120 | 0.354825 | 0.758261 | 0.374214 | 0.300688 | 2.196e-06/2.196e-06/4.392e-05 | 24.70 | 18.47 | 30.57 | 3.074/3.227 |
| 17 | 0.226119 | 0.363919 | 0.755346 | 0.371174 | 0.312751 | 1.611e-06/1.611e-06/3.221e-05 | 20.96 | 17.09 | 28.29 | 3.074/3.227 |
| 18 | 0.220007 | 0.365907 | 0.755564 | 0.374254 | 0.310266 | 1.135e-06/1.135e-06/2.271e-05 | 24.04 | 16.99 | 28.12 | 3.074/3.227 |
| 19 | 0.216727 | 0.365996 | 0.754104 | 0.377695 | 0.318465 | 7.861e-07/7.861e-07/1.572e-05 | 23.80 | 17.42 | 28.83 | 3.074/3.227 |
| 20 | 0.213976 | 0.369875 | 0.753982 | 0.373831 | 0.313802 | 5.722e-07/5.722e-07/1.144e-05 | 23.96 | 17.41 | 28.82 | 3.074/3.227 |

## Best-epoch per-class metrics

| Class | AUROC | AUPRC | F1@0.5 | Positive Studies | Negative Studies | TP | FP | TN | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Enlarged Cardiomediastinum | 0.677021 | 0.054202 | 0.000000 | 24 | 835 | 0 | 0 | 835 | 24 |
| Pneumothorax | 0.753041 | 0.154137 | 0.048780 | 37 | 822 | 1 | 3 | 819 | 36 |
| Consolidation | 0.758664 | 0.164582 | 0.000000 | 46 | 813 | 0 | 0 | 813 | 46 |
| Pneumonia | 0.682048 | 0.168617 | 0.000000 | 68 | 791 | 0 | 0 | 791 | 68 |
| Edema | 0.859391 | 0.413396 | 0.373626 | 115 | 744 | 34 | 33 | 711 | 81 |
| Cardiomegaly | 0.768188 | 0.437200 | 0.251046 | 177 | 682 | 30 | 32 | 650 | 147 |
| Atelectasis | 0.770410 | 0.444790 | 0.280303 | 193 | 666 | 37 | 34 | 632 | 156 |
| Lung Opacity | 0.701626 | 0.412436 | 0.183824 | 214 | 645 | 25 | 33 | 612 | 189 |
| Pleural Effusion | 0.895181 | 0.773797 | 0.694505 | 260 | 599 | 158 | 37 | 562 | 102 |
| No Finding | 0.864822 | 0.817439 | 0.724832 | 306 | 553 | 216 | 74 | 479 | 90 |
