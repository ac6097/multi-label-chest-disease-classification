# R5 actual-runtime forward/backward preflight

- R3 runtime reused: image budget 8, accumulation 4, channels-last True, checkpointing False.
- First accumulation window: 4 microbatches, 31 images / 17 Studies.
- Expected optimizer updates per full epoch from Study sampler: 1005.
- Study Logit Mean max absolute difference against explicit per-Study mean: 2.38e-07.
- Trainable groups have nonzero gradients: True; frozen modules have no gradients: True.
- Frozen weights unchanged: True; no optimizer step was executed in this preflight.
- Learning rates checked: `{'backbone': 1e-05, 'head': 0.0002}`.
- This disposable backward preflight is excluded from the formal R5 epoch and update counts.
- Test split was not read.

