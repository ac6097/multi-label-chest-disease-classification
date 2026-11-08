# H3-B L2-SP implementation audit

审计对象：与 H3 完全相同的 L2-SP 定义只把 λ 改为 0.1×H3；本次从原始 DINOv3 预训练权重独立训练。

## 关键参数

- λ_H3 = **28927799.4258**；λ_H3B = **2892779.94258**；比值 = 0.100000000000（目标 0.1）。
- 正则化范围：Stage3/4 中具有预训练对应关系的可训练参数，共 **26,583,552** 个元素 / 116 个 tensor。
- 参数命名空间：`backbone.model.downsample_layers.2, backbone.model.downsample_layers.3, backbone.model.stages.2, backbone.model.stages.3`。
- 归一化：R_SP = mean_j (θ_j − θ0_j)² / N_elements，与 H3 完全一致；分类头、冻结参数、最终 LayerNorm 不进入正则项。

## 9 项检查

| Check | Passed |
|---|---|
| only_stage3_stage4_trainable_pretrained_params | True |
| every_regularized_param_has_pretrained_counterpart | True |
| regularized_element_count_matches_H3 | True |
| pretrained_values_unchanged | True |
| initial_penalty_zero | True |
| penalty_is_element_mean | True |
| analytic_gradient_formula_matches_autograd | True |
| lambda_H3B_is_exact_0.1x_H3 | True |
| no_double_lambda_or_microbatch_amplification | True |
| stage3_and_stage4_gradient_present | True |
| classifier_excluded_from_penalty | True |
| frozen_params_unchanged_by_combined_step | True |
| classifier_updated_by_bce | True |
| adamw_weight_decay_is_0.01 | True |
| optimizer_lrs_match_R5 | True |

## 数值证据

- 初始化时正则项：0（必须为 0；预训练起点梯度也为 0，因此审计在 3 次 BCE-only update 造成非零漂移后执行）。
- BCE warmup 后正则项：3.33167e-10；独立 element-mean 手算：3.33167e-10（元素数 26,583,552）。
- 解析梯度公式 vs autograd 最大误差：0。
- 梯度累积隔离：4-microbatch 窗口（L2-SP 只反传一次）与“BCE 累积 + 单次 λ·R_SP 梯度”参考之间的最大差 0；单次 λ·R_SP 梯度范数 0.0205，因此不存在按 microbatch 数量重复放大。
- 漂移点梯度范数：BCE 27.66；单位 λ 的 R_SP 8.68e-09；H3-B λ 下 0.02511（约为 BCE 的 0.0009 倍）；H3 λ 下 0.2511。
- AdamW 参数组：backbone lr=1e-05 wd=0.01；head lr=0.0002 wd=0.01

## 结论

- 实现与 H3 相同，未发现重复 λ、错误归一化或梯度累积放大；H3-B 与 H3 的唯一差异是 λ 从 2.89278e7 降为 2.89278e6，正则化范围与 分母定义保持不变。
- 在预训练起点 L2-SP 梯度为 0 属于定义性质；本次审计在真实非零漂移点验证了梯度公式、梯度流与累积逻辑。
- 未读取 Test 标签。

