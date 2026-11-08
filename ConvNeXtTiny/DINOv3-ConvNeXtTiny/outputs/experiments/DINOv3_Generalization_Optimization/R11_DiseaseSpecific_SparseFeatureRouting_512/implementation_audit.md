# R11 implementation audit

## Feature shapes (measured, not assumed)

- Stage3: **[384, 32, 32]** at 512×512 input (measured from H2 backbone).
- Stage4: **[768, 16, 16]**; H2 pooled feature: 768.
- Channel Top-K budget K_c = ceil(0.5×384) = 192; spatial Top-K budget K_s = ceil(0.5×32×32) = 512.

## Model structure

- Frozen H2 base (backbone + original Global-only classifier) loaded from `H2_Single_Study_Loss_Weight/best.pth`; eval mode, no grads.
- 10 independent branches, each: Channel Selector MLP(384→64→384), Spatial Selector Conv1×1(384→1), STE Top-K masks, projection Conv1×1(384→64), LayerNorm, residual head 64→32→1 with zero-initialized last layer.
- Branch parameters: [76866, 76866, 76866, 76866, 76866, 76866, 76866, 76866, 76866, 76866] each; total 768,660; with frozen base 28,596,478.
- Checkpoint estimate: 109.1 MB (frozen H2 state + branches), below the 500 MB limit.

## Gradient / sparse routing tests

| Check | Passed |
|---|---|
| base_frozen | True |
| base_eval | True |
| base_state_unchanged_after_forward | True |
| base_logits_identical_to_H2_model | True |
| base_logits_identical_to_trained_classifier | True |
| zero_residual_initialization | True |
| topk_counts_and_zero_outside_mask | True |
| selector_ste_gradients_nonzero_finite | True |
| head_gradient_on_first_update | True |
| selector_gradient_after_one_step | True |
| gradient_isolation_all_10 | True |
| gate_isolation_all_10 | True |
| synthetic_routing_learnable | True |
| checkpoint_under_500MB | True |

## Synthetic routing learnability

- Branch residual fitting: channel task loss 1.1111 → 0.025686; spatial task loss 1.1111 → 0.044975.
- Direct selector capacity: channel-mask overlap with true support 1.000; spatial-mask overlap 1.000.
- Both selectors can learn a known support under direct supervision, and the residual branch trains; the STE routes gradients to the selectors rather than detaching them.

## Notes

- Hard masks are used in forward; backward uses the sigmoid soft surrogate with the standard STE formula `M_hard.detach() - M_soft.detach() + M_soft`.
- Dense tensor × mask means logical sparsity only; this implementation does not claim FLOP reduction.
- The frozen H2 model is never updated; gate=0 reproduces its image logits exactly.
- No Test labels read.
- STE details: forward uses the hard Top-K mask. Backward uses
  `M = M_hard.detach() - M_soft.detach() + M_soft` with
  `M_soft = sigmoid(score / T)` and `T=1.0`. The channel surrogate is additionally
  multiplied by C=384 before the STE subtraction/addition; this only scales the
  backward gradient (forward is still the hard mask) to avoid gradient dilution
  seen in the synthetic selector-capacity test (unscaled channel gradient was
  ~1e-6). The spatial surrogate is not scaled.

