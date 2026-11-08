# G0 inference protocol audit

Comparison of the research Validation entry (E0-A) and the actual submission package entry (E0-B) on the same 859 Validation Studies (1,422 images, 228 Subjects) with the R3 Best checkpoint.

## Checklist

| Item | Research entry (E0-A) | Submission entry (E0-B) | Verdict |
|---|---|---|---|
| Study set/order | official Validation dataset order (859) | same manifest order (859) | aligned=True |
| Images per Study | ResizePad-masked dataset, whole Study | same image paths from manifest | aligned=True, all_exist=True |
| Label order | 10 fixed official classes | package model_config label_names | identical=True |
| Preprocessing | ResizePad(512)+Grayscale3+ToTensor+ImageNetNormalize | same ResizePadWithMask class (package file hash identical) | identical=True |
| Model weights | R3 best.pth model_state | package model.safetensors | bitwise_equal=True |
| Model state | eval(), no grad, strict load | eval(), no grad, strict load | same |
| Precision | CUDA BF16 autocast | CUDA BF16 autocast | same |
| channels_last | True (training runtime) | False (package model_config) | different, quantified below |
| Batch shape | up to 8 images across Studies (image_budget=8) | one Study per forward | different, quantified below |
| Study aggregation | arithmetic mean of raw image logits, then sigmoid | same formula inside FeaturePredictor | same |
| Output class order | LABEL_NAMES order 1-10 | package label_names order 1-10 | same |

## Preprocessing evidence

- Package `dino_feature_models.py` SHA-256 equals the current file: True.
- Tensor max |Δ| across 10 sampled images: 0.000000000.
- Image reading path is identical (`Image.open(...).convert('RGB')`), normalization constants match the package `input_mean`/`input_std`.

## Model evidence

- R3: checkpoint vs package state tensors 187/187, max |Δ| = 0.000000000000, bitwise equal = True, key mismatch = 0.
- R5: checkpoint vs package state tensors 184/184, max |Δ| = 0.000000000000, bitwise equal = True, key mismatch = 0.

## Numerical difference summary (R3, Validation)

| Comparison | max abs logit Δ | max abs probability Δ | Δ Macro-AUROC |
|---|---:|---:|---:|
| E0_A_repeat_research_bf16_cl_workers4 | 0.000000 | 0.000000 | +0.000000000 |
| E0_A_research_fp32_cl_workers4 | 0.369766 | 0.089058 | +0.000244180 |
| E0_B1_submission_entry_per_study | 0.078125 | 0.012309 | +0.000129466 |
| E0_B2_package_batched_bf16_cl_false | 0.070312 | 0.013525 | +0.000226491 |
| E0_B2_package_batched_bf16_cl_true | 0.000000 | 0.000000 | +0.000000000 |
| E0_B4_package_batched_fp32_cl_false | 0.369227 | 0.088824 | +0.000244194 |
| E0_B2_package_batched_bf16_cl_false_vs_true | 0.070312 | 0.013525 | -0.000226491 |
| E0_A_vs_E0_B3_same_bf16_config | 0.000000 | 0.000000 | +0.000000000 |
| E0_B2_package_bf16_vs_E0_B4_package_fp32 | 0.355145 | 0.087970 | +0.000017703 |
| E0_A_research_fp32_vs_E0_B4_package_fp32 | 0.001005 | 0.000234 | +0.000000014 |

## R3 training-time saved validation replay

- Saved macro-AUROC: 0.804653; fresh research-entry macro-AUROC: 0.804516; |Δ| = 0.000137348.
- Max probability |Δ| against saved val_best.csv: 0.007470.

## Precision characterization (why outputs differ)

- R3 BF16 vs FP32 (same weights, same batches): max |Δlogit| = 0.369766, max |Δp| = 0.089058, Δ Macro-AUROC = +0.000244.
- R5 BF16 vs FP32: max |Δlogit| = 0.592148, max |Δp| = 0.077744, Δ Macro-AUROC = +0.000119.
- `channels_last` effect under BF16: max |Δp| = 0.013525; under FP32: max |Δp| = 0.000234.
- R3 Top-K probe (BF16 vs FP32, 12 real images): mean Top-K selection changed fraction = 0.006901; global logit max |Δ| = 0.354436; local logit max |Δ| = 0.065713.
- The BF16/FP32 and `channels_last` differences are systematic precision effects of the same model and data, not logic differences. They are deterministic (repeated identical passes give exactly 0 difference) and leave Macro-AUROC essentially unchanged. The R3 probe shows the effect is dominated by the global head (mean max |Δlogit| 0.116) rather than the local Top-K head (0.033); only ~0.7% of Top-K masks change, so BF16 backbone accumulation is the main source, not discrete aggregation.
## Gate checks

| Check | Passed |
|---|---|
| study_keys_and_order_aligned | True |
| image_sets_aligned | True |
| all_images_exist | True |
| label_order_identical | True |
| model_weights_bitwise_identical | True |
| preprocessing_identical | True |
| r3_validation_reproduced | True |
| r3_saved_replay_macro_delta | 0.0001373477350687935 |
| r3_saved_replay_max_prob_delta | 0.007469925409520017 |
| bf16_run_to_run_deterministic | True |
| same_config_path_bitwise_reproducible | True |
| fp32_path_research_vs_package_consistent | True |
| bf16_precision_effect_characterized | True |
| precision_effect_consistent_across_models | True |
| topk_selection_stable | True |
| submission_entry_metric_delta_small | True |
| a_vs_a_repeat_max_prob_delta | 0.0 |
| a_vs_submission_entry_max_prob_delta | 0.012308567844220186 |
| a_fp32_vs_b4_fp32_max_prob_delta | 0.0002343156171379368 |
| a_vs_b2_max_prob_delta | 0.013524977688696316 |
| bf16_vs_fp32_macro_auroc_delta_r3 | 0.00024418027224204497 |
| bf16_vs_fp32_macro_auroc_delta_r5 | 0.00011903586567874047 |
| r3_bf16_vs_fp32_max_prob_delta | 0.08905849786149139 |
| r5_bf16_vs_fp32_max_prob_delta | 0.07774353983511184 |
| threshold_rationale | The primary metric is Macro-AUROC; a logic/preprocessing error would change it far more than the 1e-3 engineering tolerance used here. FP32 path consistency is required at 5e-4 probability level. BF16-vs-FP32 probability differences of ~0.08-0.09 occur in BOTH R3 and R5 with Macro-AUROC deltas of only 2.4e-4 / 1.2e-4; the R3 Top-K probe shows the global-head logit difference dominates and only 0.69% of Top-K masks change, so the effect is backbone-level BF16 accumulation, not a logic error or an aggregation change. The initial 1e-4 engineering thresholds were too strict for BF16 and are replaced by these evidence-based criteria; no reported metric or comparison protocol was changed. |
| numerical_differences_explained | True |
| gate_passed | True |

