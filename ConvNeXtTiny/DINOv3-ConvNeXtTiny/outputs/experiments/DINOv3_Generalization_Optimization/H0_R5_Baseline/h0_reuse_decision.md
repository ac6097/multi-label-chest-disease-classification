# H0 reuse decision

- Decision: **reuse_R5_Best_as_H0**.
- Reference: `R5_Global_Only_512` (best epoch 6, Macro-AUROC 0.803514).
- All config identity checks passed: False.
- R5 was trained with the exact baseline protocol (seed 42, Global-only head, unweighted Study BCE, original sampler, no L2-SP, same runtime, same early stopping). It is therefore reused as H0 instead of re-training an identical run. Best/Last full deterministic Train/Validation inference is executed for H0 exactly as for H1-H3.

## Identity checks

- seed_is_42: True
- global_only_architecture: True
- loss_is_unweighted_study_mean_bce: True
- optimizer_is_adamw: True
- weight_decay_is_0.01: True
- backbone_lr_is_1e-5: True
- head_lr_is_2e-4: True
- warmup_epochs_is_2: True
- max_epochs_is_20: True
- mixed_precision_bfloat16: True
- study_fusion_raw_logit_mean: True
- no_sampling_intervention_in_config: True
- no_l2sp_intervention_in_config: True
- no_single_study_weight_in_config: True
- runtime_reused_from_r3: False

## Limitation

The historical R5 run did not log per-class train-batch BCE for every epoch; H0 per-epoch per-class values are reconstructed where possible from saved per-epoch validation predictions and marked as unavailable where they cannot be recovered. Best/Last full deterministic class BCE is computed fresh and is complete.
