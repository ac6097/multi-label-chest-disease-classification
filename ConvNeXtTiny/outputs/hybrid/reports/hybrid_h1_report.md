# Hybrid H1 report

## Self-supervised stage

# Hybrid H1_PIXEL self-supervised pretraining

- Status: completed 20 epochs
- Epochs completed: 20 / 20
- Best epoch by training reconstruction loss: 20
- Best total / CNN / ViT reconstruction loss: 0.006200 / 0.005915 / 0.006484
- Encoder export for supervised fine-tuning: `\\192.168.60.1\团队文件-Public\multi-label-chest-disease-classification\ConvNeXtTiny\outputs\hybrid\h1_pretrain\encoder_best.pth`
- Best/Last full checkpoint bytes: 605,284,236/605,284,236; best/last encoder export bytes: 198,577,859/198,577,859
- Initial ImageNet checkpoint SHA-256: `d11bb1f885df2a43b8160531e593bd76cc1bbc9bb724ca51e599afcf1073e121`
- Train image manifest SHA-256: `17077eb7d204f71eaf845ba7f88050437e75a21ac01a2b9dc4f79236ee5afb2f`
- Pretraining parameters: CNN 27,864,960; ViT 21,744,000; CNN/ViT decoders 494,720/295,680; total 50,399,744.
- Data source: official train images only; disease label columns were not read; valid/test were not used.
- Target effective batch: 32 images; each epoch traverses every official train image exactly once.
- Decoder state is retained in resumable best/last pretraining checkpoints and excluded from encoder export.
- Encoder/decoder gradient and parameter-change diagnostics are recorded on the first optimizer update of each epoch; non-finite/zero gradients or collapsed features/outputs stop the run with a failure checkpoint.
- Mask: exactly 60 of 100 random 32×32 pixel regions per image; masks are keyed by Subject/Study/image ID so H1/H2 share positions even when microbatch sizes differ; the same 10×10 mask is mapped to the ViT 20×20 patch grid.
- Reconstruction target: the shared augmented RGB tensor in [0,1]; MSE is computed only on masked pixels and averaged equally across branches.
- Implementation: ordinary dense PyTorch ConvNeXtV2 receives zero-filled masked pixels. The official sparse FCMAE encoder was not used; dense convolution and visible-neighbor context remain, so this is not a claim of sparse-FCMAE equivalence.
- Visuals show original target, shared masked input, and both branch reconstructions.

## Epoch log

| Epoch | Total reconstruction loss | CNN loss | ViT loss | Mask ratio | Images/s | Peak allocated/reserved GiB | Encoder gradient norms |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 0.139835 | 0.104678 | 0.174992 | 0.600 | 20.95 | 1.920/2.191 | CNN=3.872, ViT=6.89, decoders=0.1987/0.7477 |
| 2 | 0.023453 | 0.018624 | 0.028283 | 0.600 | 22.16 | 2.116/2.459 | CNN=2.288, ViT=0.3053, decoders=0.01488/0.007396 |
| 3 | 0.015094 | 0.011376 | 0.018812 | 0.600 | 23.17 | 1.925/2.459 | CNN=0.3641, ViT=0.2811, decoders=0.009963/0.009108 |
| 4 | 0.012194 | 0.009774 | 0.014614 | 0.600 | 22.46 | 1.925/2.459 | CNN=0.3397, ViT=0.1042, decoders=0.00978/0.001657 |
| 5 | 0.010583 | 0.008882 | 0.012284 | 0.600 | 21.90 | 1.925/2.459 | CNN=0.2269, ViT=0.1353, decoders=0.008622/0.004729 |
| 6 | 0.009528 | 0.008208 | 0.010849 | 0.600 | 21.90 | 1.925/2.459 | CNN=0.3632, ViT=0.09317, decoders=0.02926/0.003424 |
| 7 | 0.008699 | 0.007639 | 0.009760 | 0.600 | 22.29 | 1.925/2.459 | CNN=0.1468, ViT=0.1002, decoders=0.01118/0.00328 |
| 8 | 0.008156 | 0.007296 | 0.009016 | 0.600 | 21.69 | 1.925/2.459 | CNN=0.2421, ViT=0.1428, decoders=0.009313/0.0041 |
| 9 | 0.007670 | 0.006935 | 0.008406 | 0.600 | 21.93 | 1.925/2.459 | CNN=0.1024, ViT=0.0589, decoders=0.006599/0.001735 |
| 10 | 0.007340 | 0.006737 | 0.007943 | 0.600 | 22.05 | 1.925/2.459 | CNN=0.1029, ViT=0.04011, decoders=0.005738/0.001854 |
| 11 | 0.007079 | 0.006561 | 0.007597 | 0.600 | 22.14 | 1.925/2.459 | CNN=0.125, ViT=0.09573, decoders=0.005659/0.003799 |
| 12 | 0.006901 | 0.006449 | 0.007353 | 0.600 | 21.99 | 1.925/2.459 | CNN=0.2247, ViT=0.07048, decoders=0.005557/0.002974 |
| 13 | 0.006683 | 0.006281 | 0.007085 | 0.600 | 22.22 | 1.925/2.459 | CNN=0.1243, ViT=0.04219, decoders=0.007758/0.001848 |
| 14 | 0.006556 | 0.006194 | 0.006918 | 0.600 | 21.91 | 1.925/2.459 | CNN=0.1369, ViT=0.05082, decoders=0.007092/0.0009251 |
| 15 | 0.006452 | 0.006113 | 0.006790 | 0.600 | 21.95 | 1.925/2.459 | CNN=0.05657, ViT=0.03079, decoders=0.00491/0.001131 |
| 16 | 0.006354 | 0.006033 | 0.006676 | 0.600 | 22.57 | 1.925/2.459 | CNN=0.1556, ViT=0.05315, decoders=0.006755/0.001315 |
| 17 | 0.006310 | 0.006011 | 0.006609 | 0.600 | 22.08 | 1.925/2.459 | CNN=0.07217, ViT=0.05236, decoders=0.004853/0.001369 |
| 18 | 0.006246 | 0.005949 | 0.006544 | 0.600 | 22.17 | 1.925/2.459 | CNN=0.1124, ViT=0.02905, decoders=0.006701/0.0008616 |
| 19 | 0.006221 | 0.005939 | 0.006504 | 0.600 | 22.51 | 1.925/2.459 | CNN=0.07164, ViT=0.03119, decoders=0.003082/0.0008563 |
| 20 | 0.006200 | 0.005915 | 0.006484 | 0.600 | 22.25 | 1.925/2.459 | CNN=0.05236, ViT=0.03211, decoders=0.002714/0.0003961 |


## Classification fine-tuning

# Hybrid H1 classification report

- Status: completed 20 epochs
- Best Epoch: 6
- Best Validation Macro-AUROC: 0.793342
- Best Train Loss / Valid Loss: 0.291856 / 0.313745
- Best Macro-AUPRC / Macro-F1: 0.416213 / 0.271006
- Best checkpoint: `\\192.168.60.1\团队文件-Public\multi-label-chest-disease-classification\ConvNeXtTiny\outputs\hybrid\h1\best.pth`
- Last checkpoint: `\\192.168.60.1\团队文件-Public\multi-label-chest-disease-classification\ConvNeXtTiny\outputs\hybrid\h1\last.pth`
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
| 1 | 0.393049 | 0.365806 | 0.708944 | 0.315334 | 0.092815 | 4.968e-09/4.968e-09/9.935e-08 | 22.13 | 8.63 | 14.28 | 3.074/3.170 |
| 2 | 0.339203 | 0.342285 | 0.745951 | 0.350899 | 0.174856 | 4.998e-06/4.998e-06/9.995e-05 | 23.97 | 17.29 | 28.62 | 3.074/3.205 |
| 3 | 0.323169 | 0.328927 | 0.773216 | 0.377575 | 0.226821 | 1.000e-05/1.000e-05/2.000e-04 | 24.07 | 17.02 | 28.17 | 3.074/3.227 |
| 4 | 0.310735 | 0.339541 | 0.781379 | 0.393464 | 0.238858 | 9.928e-06/9.928e-06/1.986e-04 | 24.44 | 17.70 | 29.31 | 3.074/3.227 |
| 5 | 0.301730 | 0.317600 | 0.784751 | 0.401910 | 0.246102 | 9.713e-06/9.713e-06/1.943e-04 | 22.88 | 17.60 | 29.13 | 3.074/3.227 |
| 6 | 0.291856 | 0.313745 | 0.793342 | 0.416213 | 0.271006 | 9.363e-06/9.363e-06/1.873e-04 | 24.26 | 17.37 | 28.75 | 3.074/3.227 |
| 7 | 0.281260 | 0.326195 | 0.780480 | 0.407418 | 0.251000 | 8.889e-06/8.889e-06/1.778e-04 | 24.39 | 17.68 | 29.26 | 3.074/3.227 |
| 8 | 0.269738 | 0.343259 | 0.779599 | 0.406040 | 0.279462 | 8.303e-06/8.303e-06/1.661e-04 | 24.28 | 17.22 | 28.51 | 3.074/3.227 |
| 9 | 0.255611 | 0.334979 | 0.775263 | 0.398538 | 0.297805 | 7.624e-06/7.624e-06/1.525e-04 | 24.23 | 17.58 | 29.11 | 3.074/3.227 |
| 10 | 0.239981 | 0.341887 | 0.773958 | 0.399435 | 0.331492 | 6.874e-06/6.874e-06/1.375e-04 | 24.24 | 17.64 | 29.20 | 3.074/3.227 |
| 11 | 0.223482 | 0.356994 | 0.773316 | 0.396871 | 0.324856 | 6.075e-06/6.075e-06/1.215e-04 | 24.39 | 16.93 | 28.03 | 3.074/3.227 |
| 12 | 0.207293 | 0.367057 | 0.768481 | 0.389237 | 0.335009 | 5.250e-06/5.250e-06/1.050e-04 | 23.96 | 17.07 | 28.26 | 3.074/3.227 |
| 13 | 0.191821 | 0.382379 | 0.764001 | 0.384348 | 0.341854 | 4.426e-06/4.426e-06/8.851e-05 | 24.08 | 17.43 | 28.86 | 3.074/3.227 |
| 14 | 0.177906 | 0.399367 | 0.759679 | 0.383048 | 0.337445 | 3.626e-06/3.626e-06/7.252e-05 | 23.69 | 17.28 | 28.60 | 3.074/3.227 |
| 15 | 0.165171 | 0.409462 | 0.762452 | 0.380226 | 0.344359 | 2.874e-06/2.874e-06/5.748e-05 | 24.17 | 17.90 | 29.64 | 3.074/3.227 |
| 16 | 0.155278 | 0.420373 | 0.758220 | 0.379323 | 0.341109 | 2.196e-06/2.196e-06/4.392e-05 | 24.67 | 18.54 | 30.70 | 3.074/3.227 |
| 17 | 0.147936 | 0.431277 | 0.759918 | 0.379848 | 0.348101 | 1.611e-06/1.611e-06/3.221e-05 | 24.70 | 18.29 | 30.28 | 3.074/3.227 |
| 18 | 0.140877 | 0.438452 | 0.758153 | 0.378500 | 0.347313 | 1.135e-06/1.135e-06/2.271e-05 | 25.26 | 18.92 | 31.32 | 3.074/3.227 |
| 19 | 0.136943 | 0.440622 | 0.757282 | 0.377589 | 0.347623 | 7.861e-07/7.861e-07/1.572e-05 | 24.83 | 18.15 | 30.05 | 3.074/3.227 |
| 20 | 0.132947 | 0.451408 | 0.758202 | 0.376995 | 0.348165 | 5.722e-07/5.722e-07/1.144e-05 | 24.71 | 18.49 | 30.61 | 3.074/3.227 |

## Best-epoch per-class metrics

| Class | AUROC | AUPRC | F1@0.5 | Positive Studies | Negative Studies | TP | FP | TN | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Enlarged Cardiomediastinum | 0.705664 | 0.070014 | 0.000000 | 24 | 835 | 0 | 0 | 835 | 24 |
| Pneumothorax | 0.819902 | 0.266803 | 0.052632 | 37 | 822 | 1 | 0 | 822 | 36 |
| Consolidation | 0.767675 | 0.157588 | 0.000000 | 46 | 813 | 0 | 0 | 813 | 46 |
| Pneumonia | 0.722252 | 0.199353 | 0.000000 | 68 | 791 | 0 | 0 | 791 | 68 |
| Edema | 0.879342 | 0.501742 | 0.460829 | 115 | 744 | 50 | 52 | 692 | 65 |
| Cardiomegaly | 0.782975 | 0.455421 | 0.345865 | 177 | 682 | 46 | 43 | 639 | 131 |
| Atelectasis | 0.779415 | 0.478355 | 0.298851 | 193 | 666 | 39 | 29 | 637 | 154 |
| Lung Opacity | 0.712168 | 0.433548 | 0.094017 | 214 | 645 | 11 | 9 | 636 | 203 |
| Pleural Effusion | 0.899043 | 0.773014 | 0.734463 | 260 | 599 | 195 | 76 | 523 | 65 |
| No Finding | 0.864985 | 0.826287 | 0.723404 | 306 | 553 | 204 | 54 | 499 | 102 |
