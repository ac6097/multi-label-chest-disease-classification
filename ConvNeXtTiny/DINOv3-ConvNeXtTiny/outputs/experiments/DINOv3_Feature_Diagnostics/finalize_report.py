from __future__ import annotations

import json
import hashlib
import platform
from pathlib import Path

import pandas as pd
from PIL import Image
import sklearn
import torchvision

import run_diagnostics as diagnostics


OUT = Path(__file__).resolve().parent


def main():
    read_json = lambda name: json.loads((OUT / name).read_text(encoding="utf-8"))
    replay = read_json("r3_checkpoint_reproduction.json")
    local_global = pd.read_csv(OUT / "r3_local_global_disease_comparison.csv")
    response = pd.read_csv(OUT / "r3_response_statistics.csv")
    response_corr = pd.read_csv(OUT / "r3_disease_map_correlations.csv")
    samples = pd.read_csv(OUT / "r3_response_samples.csv")
    logits = pd.read_csv(OUT / "r3_global_vs_local_logits.csv")
    report_stats = local_global.merge(
        response[["disease", "max_topk_pixel_share_in_padding"]],
        on="disease", how="left", validate="one_to_one")
    r2_replay = read_json("r2_checkpoint_reproduction.json")["result"]
    correlations = []
    pairs = (("A_R3_fused", "B_R3_global"),
             ("B_R3_global", "C_R2_gap"),
             ("A_R3_fused", "C_R2_gap"))
    for _, row in logits.iterrows():
        for left, right in pairs:
            suffix = f"{left}_{right}"
            correlations.append({
                "left": left,
                "right": right,
                "disease": row["disease"],
                "pearson_prediction_correlation": row[
                    f"pearson_prediction_correlation_{suffix}"],
                "spearman_prediction_correlation": row[
                    f"spearman_prediction_correlation_{suffix}"],
            })
    multiview = OUT / "r3_response_visualizations" / "multiview_same_study_diagnostics.png"
    multiview_count = 0
    if multiview.is_file():
        with Image.open(multiview) as image:
            multiview_count = round(image.height / 600)
    metadata = replay["checkpoint_runtime"]
    training = diagnostics.training_comparison()
    resolution_metrics = pd.read_csv(OUT / "resolution_per_class_metrics.csv")
    resolution_bootstrap = read_json("resolution_paired_bootstrap.json")
    alignment = read_json("input_alignment_audit.json")
    r2_replay = read_json("r2_checkpoint_reproduction.json")["result"]
    diagnostics.write_resolution_report(
        OUT,
        resolution_metrics,
        resolution_bootstrap,
        training,
        alignment,
        r2_replay,
        replay["result"],
        {run: resolution_metrics[f"{run}_auc"].to_numpy()
         for run in ("R0", "R1", "R2")},
    )
    diagnostics.write_ablation_report(
        OUT,
        pd.read_csv(OUT / "r3_inference_ablation.csv"),
        read_json("r3_inference_bootstrap.json"),
        logits,
        pd.DataFrame(correlations),
        r2_replay,
        replay["result"],
        metadata,
    )
    diagnostics.write_response_report(
        OUT,
        response,
        response_corr,
        samples,
        {"selected_visualizations": len(samples), "multiview_studies": multiview_count},
        metadata,
    )
    diagnostics.write_final_report(
        OUT,
        pd.read_csv(OUT / "resolution_per_class_metrics.csv"),
        read_json("resolution_paired_bootstrap.json"),
        pd.read_csv(OUT / "r3_inference_ablation.csv"),
        read_json("r3_inference_bootstrap.json"),
        report_stats,
        training,
        metadata,
        replay["result"],
    )
    valid_csv = diagnostics.VALID_CSV
    digest = hashlib.sha256()
    with valid_csv.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    r2_runtime = read_json("r2_checkpoint_reproduction.json")["checkpoint_runtime"]
    metadata = {
        "scope": "official Validation only",
        "train_data_accessed": False,
        "test_data_accessed": False,
        "training_started": False,
        "optimizer_or_backward_step": False,
        "validation_csv": str(valid_csv),
        "validation_csv_sha256": digest.hexdigest(),
        "validation_studies": 859,
        "validation_subjects": 228,
        "validation_images": 1422,
        "label_semantics": "closed_world",
        "label_order": diagnostics.LABELS,
        "study_fusion": "arithmetic mean of raw image logits, then sigmoid",
        "bootstrap": {
            "replicates": diagnostics.BOOTSTRAP_REPLICATES,
            "seed": diagnostics.BOOTSTRAP_SEED,
            "sampling_indices_sha256": read_json("r3_inference_bootstrap.json")[
                "sampling_indices_sha256"],
        },
        "environment": {
            "python": platform.python_version(),
            "torch": diagnostics.torch.__version__,
            "torchvision": torchvision.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "cuda_runtime": diagnostics.torch.version.cuda,
            "gpu": diagnostics.torch.cuda.get_device_name(0),
            "bf16_autocast": bool(diagnostics.torch.cuda.is_bf16_supported()),
        },
        "r2_checkpoint_replay": r2_replay,
        "r3_checkpoint_replay": replay["result"],
        "r2_checkpoint_runtime": r2_runtime,
        "r3_checkpoint_runtime": metadata,
        "resolution_training_runtime": training.to_dict(orient="records"),
    }
    (OUT / "diagnostic_run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
