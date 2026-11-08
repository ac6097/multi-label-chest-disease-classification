# H0 reuse report (R5 Global-only baseline)

- H0 is the historical R5 Seed=42 baseline, reused because every config identity check passed (`h0_reuse_decision.json`).
- Checkpoints and training history are copied into this directory; the full deterministic Train/Validation inference was recomputed here.
- Limitation: the historical R5 run did not log per-epoch per-class train-batch BCE (marked unavailable); per-epoch validation class BCE is derived from its saved per-epoch validation predictions, while Best/Last full per-class BCE is complete.

