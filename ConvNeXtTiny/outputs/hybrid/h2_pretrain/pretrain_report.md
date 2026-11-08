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
