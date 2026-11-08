from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score

import run_diagnostics as diagnostics


OUT = Path(__file__).resolve().parent


def ap_vector(labels, scores):
    return np.asarray([
        average_precision_score(labels[:, i], scores[:, i])
        if np.any(labels[:, i] == 1) else np.nan
        for i in range(labels.shape[1])
    ], dtype=np.float64)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to recover R3 Global-only study predictions")
    device = torch.device("cuda")
    valid_base, official, labels = diagnostics.official_dataset()
    draws, draws_sha = diagnostics.make_subject_draws(official)

    saved_r3, saved_r3_scores = diagnostics.read_predictions(
        diagnostics.PREDICTIONS["R3"], official, labels)
    saved_r2, saved_r2_scores = diagnostics.read_predictions(
        diagnostics.PREDICTIONS["R2"], official, labels)
    r3 = diagnostics.infer_r3(valid_base, device)
    r2 = diagnostics.infer_r2(valid_base, device)
    scores = {
        "A": 1.0 / (1.0 + np.exp(-r3["fused_logits"])),
        "B": 1.0 / (1.0 + np.exp(-r3["global_logits"])),
        "C": 1.0 / (1.0 + np.exp(-r2["logits"])),
    }
    replay_a = diagnostics.validate_checkpoint_reproduction(
        "R3", saved_r3_scores, scores["A"], labels)
    replay_c = diagnostics.validate_checkpoint_reproduction(
        "R2", saved_r2_scores, scores["C"], labels)
    if (replay_a["macro_auroc_abs_error"] > 1e-5
            or replay_c["macro_auroc_abs_error"] > 1e-5
            or replay_a["max_probability_abs_error"] > 0.02
            or replay_c["max_probability_abs_error"] > 0.02):
        raise RuntimeError("Checkpoint replay failed during AUPRC bootstrap; no CI was computed")

    path = OUT / "r3_inference_bootstrap.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    if draws_sha != result["sampling_indices_sha256"]:
        raise RuntimeError("AUPRC bootstrap sampling indices differ from the saved AUROC bootstrap")
    comparisons = {}
    for left, right in (("A", "B"), ("B", "C"), ("A", "C")):
        point_left = ap_vector(labels, scores[left])
        point_right = ap_vector(labels, scores[right])
        per_class = [[] for _ in diagnostics.LABELS]
        macro_deltas = []
        for indices in draws:
            y = labels[indices]
            class_delta = []
            for class_index in range(len(diagnostics.LABELS)):
                if np.unique(y[:, class_index]).size < 2:
                    continue
                l_ap = average_precision_score(y[:, class_index],
                                               scores[left][indices, class_index])
                r_ap = average_precision_score(y[:, class_index],
                                               scores[right][indices, class_index])
                delta = float(r_ap - l_ap)
                per_class[class_index].append(delta)
                class_delta.append(delta)
            if class_delta:
                macro_deltas.append(float(np.mean(class_delta)))
        if len(macro_deltas) != diagnostics.BOOTSTRAP_REPLICATES:
            raise RuntimeError(
                f"{right}-{left}: invalid Macro-AUPRC replicates {len(macro_deltas)}")
        comparisons[f"{right}_minus_{left}"] = {
            "left": left,
            "right": right,
            "macro_auprc_delta": float(np.nanmean(point_right) - np.nanmean(point_left)),
            "macro_auprc_ci95": np.quantile(macro_deltas, [0.025, 0.975]).tolist(),
            "valid_macro_replicates": len(macro_deltas),
            "per_class": {
                name: {
                    "delta": float(point_right[i] - point_left[i]),
                    "ci95": np.quantile(per_class[i], [0.025, 0.975]).tolist()
                    if per_class[i] else [None, None],
                    "valid_replicates": len(per_class[i]),
                }
                for i, name in enumerate(diagnostics.LABELS)
            },
        }
    result["auprc_bootstrap"] = {
        "metric": "Average Precision (AUPRC)",
        "unit": "Subject cluster, paired with the same draws as the AUROC bootstrap",
        "seed": diagnostics.BOOTSTRAP_SEED,
        "requested_replicates": diagnostics.BOOTSTRAP_REPLICATES,
        "sampling_indices_sha256": draws_sha,
        "one_class_resample_rule": (
            "Skip a class if a replicate lacks either positive or negative Studies; "
            "Macro-AUPRC delta is the mean over available paired classes."
        ),
        "comparisons": comparisons,
        "checkpoint_replay": {"A_R3_fused": replay_a, "C_R2_gap": replay_c},
    }
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")
    print("Wrote paired Macro/per-class AUPRC confidence intervals using the shared Subject draws.",
          flush=True)


if __name__ == "__main__":
    main()
