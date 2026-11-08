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
