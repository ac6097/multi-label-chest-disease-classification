# R6 Actual Runtime Preflight

- Passed: True; optimizer step was executed only on the disposable preflight model.
- Runtime: image budget 8, accumulation 4, channels-last True, checkpointing False.
- Preflight window: 31 images / 17 complete Studies / 4 microbatches.
- Peak allocated/reserved: 2.94/3.99 GiB.
- All trainable module groups produced nonzero gradients and changed parameters after the optimizer step.
- Formal training builds a new model from the original pinned DINOv3 weights and the same seed-42 Global Head initialization.
