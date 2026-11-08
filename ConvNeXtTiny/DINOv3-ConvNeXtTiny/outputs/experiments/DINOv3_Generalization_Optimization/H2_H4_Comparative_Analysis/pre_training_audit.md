# H2-B / H2-C / H4 pre-training audit

- Status: completed; Seed=42; input=512.

## 1. R5 Global-only classifier head

- Head parameter names/shapes: `{"classifier.weight": [10, 768], "classifier.bias": [10]}`.
- Linear layers outside backbone: ['classifier'].
- Has any Dropout: **False**; has head LayerNorm: **False**.
- Forward order: backbone.forward_features(images) -> pooled (final LayerNorm + GAP, dim 768) -> classifier Linear(768,10).
- Original head init SHA-256: `40dc9552892097293310d9fca16e61913c5fa11eae96e3d56e572553141827a9`.

## 2. H4 dropout insertion

- Dropout p = **0.1**, parent module = `dropout`, has parameters = False.
- State dict keys identical to R5: **True**; parameter counts R5/H4 = 27827818 / 27827818.
- Train mode: two forwards differ = **True**; eval mode: two forwards exactly equal = **True**; eval logits equal R5 with same weights = **True**.
- 结论：H4 在训练时对 pooled feature 施加 p=0.10 dropout，推理 `model.eval()` 自动关闭；不新增参数、不改变 checkpoint 结构。

## 3. Single-image weight / normalization

| Experiment | weight | window loss | microbatch accumulated | microbatch matches | mixture identity |
|---|---:|---:|---:|---|---|
| H0 | 1.00 | 1.121746 | 1.121747 | True | True |
| H2B | 1.10 | 1.123443 | 1.123443 | True | True |
| H2 | 1.25 | 1.125793 | 1.125793 | True | True |
| H2C | 1.50 | 1.129261 | 1.129261 | True | True |
- H2-B/H2-C 与已验收 H2 使用完全相同的归一化：`sum_s w_s ell_s / sum_s w_s`，分母只统计当前 optimizer window 内的 eligible Study（4 个 microbatch 合并），不会因 microbatch 切分产生有效权重偏差。
- H0 权重=1 时损失与未加权 Study BCE 完全一致：True。

## 4. Freeze / initialization

- total/trainable/frozen = 27,827,818 / 26,592,778 / 1,235,040; matches R5: **True**.
- Initial backbone SHA-256 matches R5: **True**.
- Stage3/4/final-norm/classifier 可训练；Stem/Stage1/Stage2 冻结，与原 R5 一致。

## 5. Checks

| Check | Passed |
|---|---|
| r5_head_has_no_dropout | True |
| r5_head_has_single_linear | True |
| r5_head_no_layernorm | True |
| h4_dropout_p_is_0.10 | True |
| h4_dropout_positioned_after_pooled_before_linear | True |
| h4_state_dict_identical_to_r5 | True |
| h4_parameter_count_unchanged | True |
| h4_dropout_active_in_train | True |
| h4_dropout_off_in_eval | True |
| h4_eval_matches_r5 | True |
| h2b_weight_1.10 | True |
| h2c_weight_1.50 | True |
| h2_microbatch_normalization | True |
| h2_weighted_mixture_identity | True |
| freeze_policy_matches_r5 | True |
| initial_weights_match_r5 | True |

**All checks passed: True**

未读取 Test 标签；测试 forward/backward 仅用于审计，不进入正式训练。

