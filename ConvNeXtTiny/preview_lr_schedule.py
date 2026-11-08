"""Generate and validate the complete optimizer-update LR schedule before training."""
import json
import math
from pathlib import Path

import pandas as pd

from datasets import StudyBatchSampler, StudyDataset
from engine.trainer import _draw_curve
from train import build_lr_schedule_table
from utils.config import PROJECT_DIR, read_yaml, resolve_config_path, resolve_data_paths


def main():
    data_path = resolve_config_path(PROJECT_DIR / "configs/data.yaml")
    train_path = resolve_config_path(
        PROJECT_DIR / "configs/train_dsap_rpb_cosine20_no_earlystop.yaml"
    )
    data_cfg = resolve_data_paths(read_yaml(data_path), data_path)
    train_cfg = read_yaml(train_path)
    dataset = StudyDataset(
        data_cfg["train_csv"], data_cfg["image_root"], transform=None,
        label_column=data_cfg["label_column"],
        label_semantics=data_cfg["label_semantics"],
        image_id_column=data_cfg["image_id_column"],
        subject_column=data_cfg["subject_column"],
        study_column=data_cfg["study_column"],
    )
    sampler = StudyBatchSampler(
        dataset, train_cfg["image_budget"], shuffle=True, seed=train_cfg["seed"]
    )
    updates_by_epoch = []
    for epoch_index in range(int(train_cfg["epochs"])):
        sampler.set_epoch(epoch_index)
        updates_by_epoch.append(
            math.ceil(len(sampler) / int(train_cfg["grad_accumulation"]))
        )
    sampler.set_epoch(0)
    spec, rows = build_lr_schedule_table(train_cfg, updates_by_epoch)
    by_epoch = {row["epoch"]: row for row in rows}
    peak_backbone = float(train_cfg["learning_rate_backbone"])
    peak_head = float(train_cfg["learning_rate_head"])
    min_backbone = float(train_cfg["min_lr_backbone"])
    min_head = float(train_cfg["min_lr_head"])
    tolerance = 1e-12
    if not math.isclose(by_epoch[2]["backbone_lr_last_used"], peak_backbone,
                        rel_tol=1e-9, abs_tol=tolerance):
        raise RuntimeError("Backbone warmup did not reach its peak in Epoch 2")
    if not math.isclose(by_epoch[2]["head_lr_last_used"], peak_head,
                        rel_tol=1e-9, abs_tol=tolerance):
        raise RuntimeError("Head warmup did not reach its peak in Epoch 2")
    if not math.isclose(by_epoch[3]["backbone_lr_start"], peak_backbone,
                        rel_tol=1e-9, abs_tol=tolerance):
        raise RuntimeError("Cosine decay does not start from the peak at Epoch 3")
    for epoch in range(3, 21):
        row = by_epoch[epoch]
        if not math.isclose(row["head_lr_last_used"] / row["backbone_lr_last_used"],
                            peak_head / peak_backbone, rel_tol=1e-9, abs_tol=tolerance):
            raise RuntimeError(f"LR group ratio changed at Epoch {epoch}")
        if epoch > 3:
            previous = by_epoch[epoch - 1]
            if row["backbone_lr_last_used"] > previous["backbone_lr_last_used"] + tolerance:
                raise RuntimeError(f"Backbone LR rebounded during cosine decay at Epoch {epoch}")
            if row["head_lr_last_used"] > previous["head_lr_last_used"] + tolerance:
                raise RuntimeError(f"Head LR rebounded during cosine decay at Epoch {epoch}")
    if not math.isclose(by_epoch[20]["backbone_lr_last_used"], min_backbone,
                        rel_tol=1e-9, abs_tol=tolerance):
        raise RuntimeError("Backbone LR did not reach its minimum on Epoch 20's last update")
    if not math.isclose(by_epoch[20]["head_lr_last_used"], min_head,
                        rel_tol=1e-9, abs_tol=tolerance):
        raise RuntimeError("Head LR did not reach its minimum on Epoch 20's last update")
    for epoch in range(21, 41):
        row = by_epoch[epoch]
        if not (math.isclose(row["backbone_lr_start"], min_backbone, abs_tol=tolerance)
                and math.isclose(row["backbone_lr_last_used"], min_backbone, abs_tol=tolerance)
                and math.isclose(row["head_lr_start"], min_head, abs_tol=tolerance)
                and math.isclose(row["head_lr_last_used"], min_head, abs_tol=tolerance)):
            raise RuntimeError(f"LR did not remain at its minimum in Epoch {epoch}")

    output_dir = (PROJECT_DIR / train_cfg["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "lr_schedule_preview.csv"
    report_path = output_dir / "lr_schedule_preview.md"
    chart_path = output_dir / "lr_schedule_preview.png"
    pd.DataFrame(rows).to_csv(table_path, index=False)
    epochs = [row["epoch"] for row in rows]
    _draw_curve(chart_path, [
        ("Backbone LR start", epochs, [row["backbone_lr_start"] for row in rows], (39, 110, 220)),
        ("Backbone LR last update", epochs, [row["backbone_lr_last_used"] for row in rows], (115, 170, 230)),
        ("Head LR start", epochs, [row["head_lr_start"] for row in rows], (220, 75, 55)),
        ("Head LR last update", epochs, [row["head_lr_last_used"] for row in rows], (236, 150, 130)),
    ], "Learning Rate (log scale)", logarithmic=True,
        markers=train_cfg["plot_epoch_markers"])

    previous_best_path = PROJECT_DIR / "outputs/experiments/ConvNeXt_DSAP_RPB/best.json"
    previous_last_path = PROJECT_DIR / "outputs/experiments/ConvNeXt_DSAP_RPB/last.json"
    previous_history_path = PROJECT_DIR / "outputs/experiments/ConvNeXt_DSAP_RPB/train_log.csv"
    previous_best = (json.loads(previous_best_path.read_text(encoding="utf-8"))
                     if previous_best_path.is_file() else None)
    previous_last = (json.loads(previous_last_path.read_text(encoding="utf-8"))
                     if previous_last_path.is_file() else None)
    previous_update_range = None
    if previous_history_path.is_file():
        previous_history = pd.read_csv(previous_history_path)
        if "optimizer_updates" in previous_history and not previous_history.empty:
            previous_update_range = (
                int(previous_history["optimizer_updates"].min()),
                int(previous_history["optimizer_updates"].max()),
            )
    lines = [
        "# DSAP-RPB Cosine20 / No Early Stopping LR Schedule Preview",
        "",
        f"- Updates per epoch (deterministic shuffled sampler): {min(updates_by_epoch)}–{max(updates_by_epoch)}; total planned updates={spec['total_steps']}",
        f"- Warmup updates: {spec['warmup_steps']}",
        f"- Cosine end update: {spec['cosine_end_step']}",
        f"- Cosine steps: {spec['cosine_steps']}",
        f"- Minimum LR: Backbone {min_backbone:.3e}; Head {min_head:.3e}; ratio={min_head/min_backbone:.1f}",
        "- Scheduler advances after each successful optimizer update; skipped GradScaler updates do not advance it.",
        "- Cosine reaches the nonzero minimum on the final optimizer update of Epoch 20; Epoch 21–40 remain flat at that minimum.",
        "",
        "| Epoch | Updates | Phase | Backbone LR start | Backbone LR last update | Head LR start | Head LR last update |",
        "|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['epoch']} | {row['optimizer_updates']} | {row['phase']} | {row['backbone_lr_start']:.8e} | "
            f"{row['backbone_lr_last_used']:.8e} | {row['head_lr_start']:.8e} | "
            f"{row['head_lr_last_used']:.8e} |"
        )
    if previous_best is not None and previous_last is not None:
        old_scheduler = previous_best.get("config", {}).get("scheduler_metadata", {})
        old_total = old_scheduler.get("total_steps")
        old_warmup = old_scheduler.get("warmup_steps")
        old_cosine_updates = (old_total - old_warmup) if old_total is not None and old_warmup is not None else None
        lines.extend([
            "",
            "## Previous Scheduler Readback",
            "",
            f"- Previous best Epoch/Macro-AUROC: {previous_best.get('epoch')} / {previous_best.get('best_macro_auroc')}",
            f"- Previous last Epoch: {previous_last.get('epoch')}",
            f"- Previous scheduler: custom LambdaLR (no explicit T_max/eta_min); effective cosine horizon after warmup={old_cosine_updates} optimizer updates, minimum LR factor=0.",
            f"- Previous completed run had {previous_update_range[0]}–{previous_update_range[1]} successful optimizer updates/epoch due to shuffled Study packing." if previous_update_range else "- Previous per-Epoch optimizer-update counts were unavailable.",
            "- Previous scheduler stepped on each successful optimizer update, after gradient accumulation; GradScaler-skipped updates did not advance it.",
            "",
            "The previous DSAP-RPB run stopped after Epoch 14. This new run is a fresh ImageNet initialization and targets 40 epochs, so total optimizer updates differ.",
        ])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSchedule chart: {chart_path}")
    print(f"Schedule CSV: {table_path}")


if __name__ == "__main__":
    main()
