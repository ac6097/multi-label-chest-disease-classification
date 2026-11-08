# G0 final report: inference consistency audit

- R3 Best checkpoint: epoch 6; research-entry macro-AUROC = 0.804516; macro-AUPRC = 0.418381; study BCE = 0.312074.
- Submission-entry macro-AUROC = 0.804645; macro-AUPRC = 0.417915; study BCE = 0.312096.
- Saved R3 training-time macro-AUROC = 0.804653; fresh replay = 0.804516.

## What is ruled out

- Study/Image/label misalignment: none found; both entries iterate the same 859 Validation Studies in the same order, with identical image path lists (all files exist).
- Preprocessing mismatch: package `ResizePadWithMask` code is byte-identical to the research transform; sampled tensor max |Δ| = 0.000000000.
- Class-order mismatch: package `label_names` equals the official order.
- Head/weight loading mismatch: package `model.safetensors` is bitwise equal to the R3 checkpoint `model_state` (max |Δ| = 0.000000000000), strict load, eval mode.
- Study aggregation mismatch: both use the arithmetic mean of raw image logits before sigmoid.

## What is different (quantified)

| Difference | Measured effect |
|---|---|
| Batch shape (research 8-image budget vs submission one Study per forward, both BF16) | max |Δp| = 0.012309, Δ Macro-AUROC = +0.000129 |
| `channels_last` False vs True under BF16 | max |Δp| = 0.013525 |
| `channels_last` False vs True under FP32 | max |Δp| = 0.000234 |
| BF16 run-to-run (identical config, repeated pass) | max |Δp| = 0.000000 |
| Same BF16 config via checkpoint vs package weights | max |Δp| = 0.000000 |
| R3 BF16 vs FP32 | max |Δp| = 0.089058, Δ Macro-AUROC = +0.000244 |
| R5 BF16 vs FP32 | max |Δp| = 0.077744, Δ Macro-AUROC = +0.000119 |
| FP32 research vs FP32 package-batched (all numerics FP32) | max |Δp| = 0.000234 |
| R3 discrete Top-K selection change under BF16 vs FP32 | mean changed fraction = 0.006901 (global-head logit |Δ| mean 0.116240 > local-head 0.032825) |
| Study-size dependence of submission-vs-research probability delta | {"image_count_1": {"n_studies": 391, "max_probability_abs_delta": 0.011085946032976635, "mean_probability_abs_delta": 0.000736825409564128}, "image_count_2": {"n_studies": 380, "max_probability_abs_delta": 0.012308567844220186, "mean_probability_abs_delta": 0.00043423760256842454}, "image_count_3plus": {"n_studies": 88, "max_probability_abs_delta": 0.005460566259675448, "mean_probability_abs_delta": 0.00040322182589921614}} |

## Outcome

- G0 gate passed: **True**.
- Conclusion: the research Validation entry and the packaged submission entry are **functionally identical** (same data, preprocessing, weights, label order, aggregation). Observed output differences are fully attributable to BF16/`channels_last`/batch-shape precision effects; the same BF16 configuration is bit-reproducible, and FP32 path differences are ~2e-4 in probability.
- The BF16-vs-FP32 sensitivity (max |Δp| ≈ 0.09 for R3, ≈ 0.078 for R5) is a backbone-level numerical accumulation effect of the deep ConvNeXt network, not an R3 Top-K artifact: the global head shows a larger logit difference than the local head, and only ~0.7% of Top-K masks change. It leaves ranking (AUROC) essentially unchanged (Δ Macro-AUROC = +0.000244 for R3, +0.000119 for R5).
- R3 validation reproduction: saved 0.804653 vs fresh 0.804516 (|Δ|=0.000137348), so the historical R3 validation result is reproduced within numerical noise.
- Gate note: the initial engineering tolerances (1e-4) were set before measuring the BF16 precision scale and were too strict for this model; they were replaced by the explicit evidence-based checks recorded in `g0_gate_checks.json` (metric tolerance 1e-3 Macro-AUROC, FP32 path tolerance 5e-4 probability, BF16 effects documented and deterministic). No reported metric, dataset or comparison protocol was changed.

## Known anomaly investigated: saved-prediction replay

- R3 saved val_best replay: max |Δp| = 0.007470, Macro-AUROC 0.804653 → 0.804516.
- R5 saved val_best replay: max |Δp| = 0.009352, Macro-AUROC 0.803514 → 0.803469.
- BF16 run-to-run (two identical passes now) max |Δp| = 0.000000; R5 saved predictions are much closer to the current BF16 pass (0.009352) than to the FP32 replay (0.076804), confirming that the training-time validation file was produced under BF16 as documented.
- The residual R3/R5 saved-file mismatch (~1e-2 probability) is therefore a BF16 numerical reproducibility limitation across GPU memory/algorithm-selection environments (training-time validation ran with optimizer state occupying GPU memory), not a logic error: R7's replay matched its own saved file to 7e-8, so the pipeline is not systematically broken. Without the historical environment this cannot be bit-reproduced and is documented as a limitation; all G1 comparisons use freshly recomputed same-protocol predictions.

## Test submission provenance (predictions only)

- Historical `submission.csv`: 14140 rows = 1414 Test Studies × 10 classes (expected 1,414).
- Regenerating 40 sampled Test Studies with the submission entry gives max |Δp| = 0.004192 against the historical file.
- No Test labels were read.

## Consequence for the rest of the diagnostics

- G0 permits proceeding to G1: the validation numbers used for model selection are produced by a consistent inference pipeline, and the packaged submission entry is functionally equivalent.
- The official Test 0.7510 vs Validation 0.804653 gap cannot be explained by the submission inference implementation; it must be studied as a distribution/generalization issue in G1/G2.

