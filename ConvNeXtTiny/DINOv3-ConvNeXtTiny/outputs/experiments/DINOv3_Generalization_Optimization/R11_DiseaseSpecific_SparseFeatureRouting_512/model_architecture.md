# R11 model architecture

- Frozen H2 Global-only base: DINOv3 ConvNeXt-Tiny, 512×512, Stem/Stage1/Stage2 frozen (H2 checkpoint), Stage3/Stage4/original classifier frozen too.
- Feature shapes measured from the actual H2 backbone: Stage3 = 384×32×32, Stage4 = 768×16×16; pooled feature = 768.
- Ten independent disease branches; for disease k:
  1. Channel selector: GAP(Stage3) → Linear(384,64) → GELU → Linear(64,384) → channel scores; Top-K K_c=192 with STE hard mask.
  2. Spatial selector: Conv1×1(384→1) → spatial scores; Top-K K_s=512 with STE hard mask.
  3. F~ = F3 ⊙ M_c ⊙ M_s (broadcast [B,C,1,1] and [B,1,H,W]).
  4. Projection Conv1×1(384→64) → masked mean pooling over selected spatial positions (denominator K_s) → LayerNorm(64).
  5. Residual head: Linear(64→32) → GELU → Linear(32→1), zero-initialized.
- Final logits: z_k' = z_k^H2 + g_k · r_k; g_k is an independent gate.
- Only branches are trainable; H2 parameters receive no gradients.
- Branch parameters: [76866, 76866, 76866, 76866, 76866, 76866, 76866, 76866, 76866, 76866] each ×10 = 768,660; frozen base 27,827,818.
- Dense tensor × mask = logical sparsity; no FLOP reduction is claimed.
- STE details: forward = hard Top-K mask; backward = sigmoid soft surrogate
  (`sigmoid(score/T)`, T=1.0) in the standard
  `M_hard.detach() - M_soft.detach() + M_soft` formula. The channel surrogate is
  multiplied by C=384 (gradient scaling only; forward unchanged) so the 384-way
  channel selector is not gradient-diluted; the spatial surrogate is unscaled.

