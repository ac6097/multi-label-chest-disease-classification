# H-series initialization verification

- Backbone: official DINOv3 ConvNeXt-Tiny, pinned local revision.
- Initial weight SHA-256: `bd30a9459d6149564ef53af6e8a1999980953b009b94cde836ac1bac4d339cb2`.
- Initial Global Head SHA-256: `40dc9552892097293310d9fca16e61913c5fa11eae96e3d56e572553141827a9`; matches R3/R5.
- No supervised-finetuned checkpoint is used; each experiment starts from the original pretrained weights.
- Parameters total/trainable/frozen: 27,827,818/26,592,778/1,235,040.
- Seed: 42; runtime: budget=8, accumulation=4, channels_last=True.

