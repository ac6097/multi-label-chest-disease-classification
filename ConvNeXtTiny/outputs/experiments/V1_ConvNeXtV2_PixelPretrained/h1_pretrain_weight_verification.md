# H1 Encoder Weight Verification

- Status: verified
- V1 source: `\\192.168.60.1\团队文件-Public\multi-label-chest-disease-classification\ConvNeXtTiny\outputs\hybrid\h1_pretrain\encoder_last.pth`
- Source role: H1 pixel-pretraining final encoder export, epoch 20
- Source SHA-256: `abf3a685177e098a3fdf628427c250f68a7e4556560dbd36ded8b63f46e643fb`
- Paired H1 full checkpoint: `\\192.168.60.1\团队文件-Public\multi-label-chest-disease-classification\ConvNeXtTiny\outputs\hybrid\h1_pretrain\last.pth` (epoch 20)
- H1 method / encoder epoch: `h1_pixel` / 20
- Shared ImageNet initialization SHA-256: `d11bb1f885df2a43b8160531e593bd76cc1bbc9bb724ca51e599afcf1073e121`
- Official Train manifest SHA-256: `17077eb7d204f71eaf845ba7f88050437e75a21ac01a2b9dc4f79236ee5afb2f`
- Validation/Test read during H1 pretraining: False / False
- Encoder export excludes pixel decoder: True
- ConvNeXtV2 state tensors loaded strictly: 196; missing: 0; unexpected: 0
- GRN state tensors verified: 36
- ConvNeXtV2 parameters: 27,864,960
- Normalization mean/std: `[0.48500001430511475, 0.4560000002384186, 0.4059999883174896]` / `[0.2290000021457672, 0.2240000069141388, 0.22499999403953552]`
- CNN-only V1 initialization export: `\\192.168.60.1\团队文件-Public\multi-label-chest-disease-classification\ConvNeXtTiny\outputs\experiments\V1_ConvNeXtV2_PixelPretrained\convnextv2_h1_encoder.pth`
- CNN-only V1 initialization SHA-256: `cce41da133713d94a147dac9d6ff7c354381c8b9d0c4004fe024f603b82e0855`
- The H1 export was inspected for provenance, then only ConvNeXtV2 tensors were extracted. ViT weights, the H1 classifier, pixel decoder, and mask tokens are excluded from the V1 model and its initialization export.
