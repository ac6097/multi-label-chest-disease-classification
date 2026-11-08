import argparse
import gc
import json
import statistics
import subprocess
import threading
import time
from pathlib import Path

import torch
import timm
import yaml
from torch.utils.data import DataLoader

from datasets import StudyBatchSampler, StudyDataset, collate_studies
from datasets.hybrid import OfficialTrainImageDataset, build_hybrid_transform
from hybrid.backbone_factory import build_hybrid_backbone
from hybrid.common import (PROJECT_DIR, autocast_context, build_adamw, current_lrs,
                           device_and_amp, read_yaml, resolve_data_config, StandardStudyBCE,
                           seed_everything, seed_worker, write_csv, write_json)
from model.hybrid import HybridClassifier
from model.hybrid_pretraining import (LatentReconstructionPretrainer,
                                      PixelReconstructionPretrainer,
                                      sample_spatial_mask_for_ids)


class NvidiaSampler:
    def __init__(self, interval=1.0):
        self.interval = float(interval)
        self.samples = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._started = False

    def _run(self):
        while not self._stop.is_set():
            try:
                result = subprocess.run([
                    "nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits",
                ], capture_output=True, text=True, timeout=5, check=True)
                fields = result.stdout.strip().splitlines()[0].split(",")
                self.samples.append((float(fields[0].strip()), float(fields[1].strip())))
            except (OSError, subprocess.SubprocessError, ValueError, IndexError):
                pass
            self._stop.wait(self.interval)

    def start(self):
        self._thread.start()
        self._started = True

    def stop(self):
        self._stop.set()
        if self._started:
            self._thread.join(timeout=6)

    def summary(self):
        if not self.samples:
            return {"gpu_utilization_mean_pct": float("nan"),
                    "gpu_memory_used_mean_mib": float("nan")}
        return {
            "gpu_utilization_mean_pct": statistics.fmean(item[0] for item in self.samples),
            "gpu_memory_used_mean_mib": statistics.fmean(item[1] for item in self.samples),
        }


def _load_shared_payload(path):
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_DIR / path
    if not path.is_file():
        raise FileNotFoundError(f"Run initialize_weights.py first: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def _datasets_for_mode(mode, data_config, image_budget, num_workers, pin_memory, seed):
    transform = build_hybrid_transform(data_config["image_size"], train=True)
    if mode == "h0_supervised":
        dataset = StudyDataset(
            data_config["train_csv"], data_config["image_root"], transform=transform,
            label_column=data_config["label_column"],
            label_semantics=data_config["label_semantics"],
            image_id_column=data_config["image_id_column"],
            subject_column=data_config["subject_column"],
            study_column=data_config["study_column"],
        )
        batch_sampler = StudyBatchSampler(dataset, image_budget, shuffle=True, seed=seed)
        generator = torch.Generator().manual_seed(seed)
        loader = DataLoader(
            dataset, batch_sampler=batch_sampler, num_workers=num_workers,
            pin_memory=pin_memory, collate_fn=collate_studies, generator=generator,
            worker_init_fn=seed_worker if num_workers else None,
            persistent_workers=bool(num_workers),
        )
        return loader, batch_sampler
    dataset = OfficialTrainImageDataset(
        data_config["train_csv"], data_config["image_root"], transform=transform,
        image_id_column=data_config["image_id_column"],
        subject_column=data_config["subject_column"],
        study_column=data_config["study_column"],
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset, batch_size=image_budget, shuffle=True, num_workers=num_workers,
        pin_memory=pin_memory, drop_last=False, generator=generator,
        worker_init_fn=seed_worker if num_workers else None,
        persistent_workers=bool(num_workers),
    )
    return loader, None


def _make_model(mode, payload, checkpointing, initial_path, initial_hash):
    backbone = build_hybrid_backbone(initial_checkpoint=initial_path,
                                     initial_checkpoint_payload=payload,
                                     initial_checkpoint_sha256=initial_hash)
    backbone.set_grad_checkpointing(checkpointing)
    if mode == "h0_supervised":
        return HybridClassifier(backbone)
    if mode == "h1_pixel":
        model = PixelReconstructionPretrainer(backbone, checkpoint_blocks=checkpointing)
    elif mode == "h2_latent":
        model = LatentReconstructionPretrainer(backbone, checkpoint_blocks=checkpointing)
    else:
        raise ValueError(f"Unknown benchmark mode {mode}")
    return model


def _parameter_groups(model, mode):
    backbone = model.backbone
    if mode == "h0_supervised":
        return [
            {"params": backbone.convnext.parameters(), "lr": 1e-5, "name": "convnext"},
            {"params": backbone.vit.parameters(), "lr": 1e-5, "name": "vit"},
            {"params": model.classifier.parameters(), "lr": 2e-4, "name": "head"},
        ]
    return [
        {"params": backbone.convnext.parameters(), "lr": 1e-5, "name": "convnext"},
        {"params": backbone.vit.parameters(), "lr": 1e-5, "name": "vit"},
        {"params": list(model.cnn_decoder.parameters()) + list(model.vit_decoder.parameters()) +
                    [model.cnn_mask_token, model.vit_mask_token]
                    if mode == "h2_latent" else
                    list(model.cnn_decoder.parameters()) + list(model.vit_decoder.parameters()) +
                    [model.vit_mask_token],
         "lr": 2e-4, "name": "decoder"},
    ]


def _fetch_window(loader, iterator, count, sampler, epoch):
    batches = []
    for _ in range(count):
        try:
            batch = next(iterator)
        except StopIteration:
            epoch += 1
            if sampler is not None:
                sampler.set_epoch(epoch)
            iterator = iter(loader)
            batch = next(iterator)
        batches.append(batch)
    return batches, iterator, epoch


def _move_and_count(mode, batch, device):
    if mode == "h0_supervised":
        valid_count = int(batch["known"].any(dim=1).sum())
        images = batch["images"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        known = batch["known"]
        study_index = batch["study_index"].to(device, non_blocking=True)
        study_keys = list(zip(batch["subject_id"], batch["study_id"]))
        subjects = list(batch["subject_id"])
        return {"images": images, "labels": labels, "known": known,
                "study_index": study_index, "num_studies": batch["num_studies"]}, \
               int(images.shape[0]), int(batch["num_studies"]), valid_count, \
               study_keys, subjects
    images = batch["image"].to(device, non_blocking=True)
    study_keys = list(zip(batch["subject_id"], batch["study_id"]))
    subjects = list(batch["subject_id"])
    return {"images": images}, int(images.shape[0]), len(set(study_keys)), \
           int(images.shape[0]), study_keys, subjects


def _candidate_update(mode, model, optimizer, window, device, amp_enabled, amp_dtype,
                      accumulation, seed, epoch, criterion=None):
    valid_count_window = sum(int(batch["known"].any(dim=1).sum()) for batch in window) if mode == "h0_supervised" else len(window)
    if mode != "h0_supervised":
        valid_count_window = sum(int(batch["image"].shape[0]) for batch in window)
    if valid_count_window <= 0:
        raise ValueError("Benchmark optimizer window contains no eligible Study/image")
    optimizer.zero_grad(set_to_none=True)
    total_images, total_studies = 0, 0
    study_keys, subjects = set(), set()
    loss_value = torch.zeros((), device=device, dtype=torch.float32)
    compute_events = []
    for batch in window:
        moved, image_count, study_count, valid_count, batch_study_keys, batch_subjects = _move_and_count(mode, batch, device)
        total_images += image_count
        total_studies += study_count
        study_keys.update(batch_study_keys)
        subjects.update(batch_subjects)
        compute_start = torch.cuda.Event(enable_timing=True)
        compute_end = torch.cuda.Event(enable_timing=True)
        compute_start.record()
        with autocast_context(device, amp_enabled, amp_dtype):
            if mode == "h0_supervised":
                logits = model(moved["images"], moved["study_index"], moved["num_studies"])
                loss = criterion(logits, moved["labels"], moved["known"])
                scale = valid_count / valid_count_window
            else:
                mask = sample_spatial_mask_for_ids(
                    batch["subject_id"], batch["study_id"], batch["image_id"],
                    epoch + 1, seed, device)
                output = model(moved["images"], mask)
                loss = output["loss"]
                scale = image_count / valid_count_window
        (loss * scale).backward()
        loss_value = loss_value + loss.detach().float() * scale
        compute_end.record()
        compute_events.append((compute_start, compute_end))
    update_start = torch.cuda.Event(enable_timing=True)
    update_end = torch.cuda.Event(enable_timing=True)
    update_start.record()
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0,
                                                  error_if_nonfinite=True)
    optimizer.step()
    update_end.record()
    torch.cuda.synchronize(device)
    compute_events.append((update_start, update_end))
    compute_seconds = sum(start.elapsed_time(end) for start, end in compute_events) / 1000.0
    return {"loss": float(loss_value.detach()), "images": total_images, "studies": total_studies,
            "study_keys": study_keys, "subjects": subjects,
            "gradient_norm": gradient_norm.detach(), "compute_seconds": compute_seconds}


def run_candidate(mode, image_budget, accumulation, checkpointing, payload, initial_path, initial_hash,
                  data_config, benchmark_config, output_dir, provenance):
    seed = int(benchmark_config["seed"])
    seed_everything(seed)
    device, amp_enabled, amp_dtype = device_and_amp(True, "bfloat16")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sampler = NvidiaSampler()
    row = {
        "mode": mode,
        "image_budget": int(image_budget),
        "grad_accumulation": int(accumulation),
        "target_effective_images": int(accumulation * image_budget),
        "num_workers": int(benchmark_config["num_workers"]),
        "pin_memory": bool(benchmark_config["pin_memory"]),
        "optimizer_fused": bool(benchmark_config["optimizer_fused"]),
        "gradient_checkpointing": bool(checkpointing),
        "warmup_updates": int(benchmark_config["warmup_updates"]),
        "measurement_updates": int(benchmark_config["measurement_updates"]),
        **provenance,
        "status": "ok",
        "oom": False,
        "error": "",
    }
    model = loader = optimizer = None
    try:
        model = _make_model(mode, payload, checkpointing, initial_path, initial_hash).to(device)
        optimizer = build_adamw(_parameter_groups(model, mode), 0.01,
                                fused=bool(benchmark_config["optimizer_fused"]))
        loader, batch_sampler = _datasets_for_mode(
            mode, data_config, image_budget, int(benchmark_config["num_workers"]),
            bool(benchmark_config["pin_memory"]), seed)
        iterator = iter(loader)
        epoch = 0
        criterion = StandardStudyBCE().to(device) if mode == "h0_supervised" else None
        step_times, compute_times, losses = [], [], []
        total_images = 0
        measured_studies, measured_subjects = set(), set()
        gradient_norms = []
        warmup_updates = int(benchmark_config["warmup_updates"])
        measured_updates = int(benchmark_config["measurement_updates"])
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        sampler.start()
        for update_index in range(warmup_updates + measured_updates):
            e2e_started = time.perf_counter()
            window, iterator, epoch = _fetch_window(loader, iterator, accumulation,
                                                   batch_sampler, epoch)
            torch.cuda.synchronize(device)
            update = _candidate_update(mode, model, optimizer, window, device,
                                       amp_enabled, amp_dtype, accumulation, seed, epoch, criterion)
            e2e_seconds = time.perf_counter() - e2e_started
            if update_index >= warmup_updates:
                step_times.append(e2e_seconds)
                compute_times.append(update["compute_seconds"])
                losses.append(update["loss"])
                total_images += update["images"]
                measured_studies.update(update["study_keys"])
                measured_subjects.update(update["subjects"])
                gradient_norms.append(update["gradient_norm"])
        torch.cuda.synchronize(device)
        sampler.stop()
        compute_total = sum(compute_times)
        e2e_total = sum(step_times)
        total_memory = torch.cuda.get_device_properties(device).total_memory / (1024 ** 2)
        peak_alloc = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        peak_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
        gpu_summary = sampler.summary()
        row.update({
            "actual_images": total_images,
            "actual_studies": len(measured_studies),
            "actual_subjects": len(measured_subjects),
            "optimizer_updates": len(step_times),
            "actual_images_per_update": total_images / len(step_times) if step_times else 0.0,
            "actual_studies_per_update": len(measured_studies) / len(step_times) if step_times else 0.0,
            "end_to_end_seconds": e2e_total,
            "gpu_compute_seconds": compute_total,
            "end_to_end_images_per_s": total_images / e2e_total if e2e_total else 0.0,
            "gpu_compute_images_per_s": total_images / compute_total if compute_total else 0.0,
            "update_p50_seconds": statistics.median(step_times),
            "update_p95_seconds": float(torch.tensor(step_times).quantile(0.95)),
            "compute_update_p50_seconds": statistics.median(compute_times),
            "compute_update_p95_seconds": float(torch.tensor(compute_times).quantile(0.95)),
            "update_cv": (statistics.pstdev(step_times) / statistics.fmean(step_times)
                          if len(step_times) > 1 and statistics.fmean(step_times) else 0.0),
            "mean_loss": statistics.fmean(losses),
            "mean_gradient_norm": float(torch.stack(gradient_norms).float().mean()),
            "peak_allocated_mib": peak_alloc,
            "peak_reserved_mib": peak_reserved,
            "peak_reserved_fraction": peak_reserved / total_memory,
            "gpu_total_memory_mib": total_memory,
            "gpu_memory_used_mean_mib": gpu_summary["gpu_memory_used_mean_mib"],
            "gpu_memory_used_mean_fraction": gpu_summary["gpu_memory_used_mean_mib"] / total_memory,
            "gpu_utilization_mean_pct": gpu_summary["gpu_utilization_mean_pct"],
            "lr_groups": current_lrs(optimizer),
        })
    except torch.cuda.OutOfMemoryError as exc:
        sampler.stop()
        row.update({"status": "oom", "oom": True, "error": str(exc)[:500]})
    except RuntimeError as exc:
        sampler.stop()
        if "out of memory" in str(exc).lower():
            row.update({"status": "oom", "oom": True, "error": str(exc)[:500]})
        else:
            row.update({"status": "error", "error": str(exc)[:1000]})
    except Exception as exc:
        sampler.stop()
        row.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"[:1000]})
    finally:
        del model, optimizer, loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_json(output_dir / f"{mode}_budget{image_budget}_accum{accumulation}_ckpt{int(checkpointing)}.json", row)
    return row


def _choose(rows, memory_margin, exact_effective_images=None, throughput_tolerance=0.03):
    valid = [row for row in rows if row["status"] == "ok" and
             row.get("end_to_end_images_per_s", 0) > 0 and
             row.get("peak_reserved_fraction", 1) <= memory_margin and
             (exact_effective_images is None or
              row.get("target_effective_images") == exact_effective_images) and
             row.get("update_cv", 1) < 0.75]
    if not valid:
        valid = [row for row in rows if row["status"] == "ok" and
                 row.get("end_to_end_images_per_s", 0) > 0 and
                 (exact_effective_images is None or
                  row.get("target_effective_images") == exact_effective_images) and
                 row.get("peak_reserved_fraction", 1) <= memory_margin]
    if not valid:
        raise RuntimeError("No stable, non-OOM benchmark configuration fits the GPU memory margin")
    top_rate = max(row["end_to_end_images_per_s"] for row in valid)
    near_top = [row for row in valid if row["end_to_end_images_per_s"] >=
                top_rate * (1.0 - throughput_tolerance)]
    return min(near_top, key=lambda row: (
        row["peak_reserved_fraction"], row["gradient_checkpointing"], row["image_budget"]))


def _write_yaml(path, config):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    temporary.replace(path)


def _candidate_result(mode, budget, accumulation, checkpointing, payload,
                      initial_path, initial_hash, data_config, benchmark_config,
                      output_root, provenance):
    candidate_path = output_root / f"{mode}_budget{budget}_accum{accumulation}_ckpt{int(checkpointing)}.json"
    expected = {
        "mode": mode,
        "image_budget": int(budget),
        "grad_accumulation": int(accumulation),
        "gradient_checkpointing": bool(checkpointing),
        "warmup_updates": int(benchmark_config["warmup_updates"]),
        "measurement_updates": int(benchmark_config["measurement_updates"]),
        "num_workers": int(benchmark_config["num_workers"]),
        "pin_memory": bool(benchmark_config["pin_memory"]),
        "optimizer_fused": bool(benchmark_config["optimizer_fused"]),
    }
    if candidate_path.is_file():
        row = json.loads(candidate_path.read_text(encoding="utf-8"))
        for key, value in expected.items():
            if row.get(key) != value:
                raise ValueError(f"Stored benchmark candidate {candidate_path.name} has mismatched {key}")
        for key, value in provenance.items():
            if key in row and row[key] != value:
                raise ValueError(f"Stored benchmark candidate {candidate_path.name} has mismatched {key}")
        row.update(provenance)
        row["gpu_name"] = torch.cuda.get_device_name(torch.device("cuda"))
        row["gpu_total_memory_mib"] = torch.cuda.get_device_properties(torch.device("cuda")).total_memory / (1024 ** 2)
        write_json(candidate_path, row)
        print(f"Reusing completed benchmark candidate {candidate_path.name}: {row['status']}")
        return row
    return run_candidate(mode, budget, accumulation, checkpointing, payload,
                         initial_path, initial_hash, data_config, benchmark_config,
                         output_root, provenance)


def run_benchmark(experiments_config_path="configs/hybrid_experiments.yaml",
                  refresh_finalization=False):
    experiments = read_yaml(experiments_config_path)
    data_config = resolve_data_config(experiments["data_config"])
    audit_path = PROJECT_DIR / "outputs/hybrid/shared/manifests/data_audit.json"
    smoke_path = PROJECT_DIR / "outputs/hybrid/smoke_test/hybrid_smoke_test.json"
    if not audit_path.is_file():
        raise FileNotFoundError("Official train/validation data audit must pass before GPU benchmarking")
    if not smoke_path.is_file() or json.loads(smoke_path.read_text(encoding="utf-8")).get("passed") is not True:
        raise RuntimeError("Hybrid real-sample Smoke Test must pass before GPU benchmarking")
    benchmark_config = read_yaml(experiments["benchmark_config"])
    output_root = Path(experiments["outputs"]["benchmark"])
    if not output_root.is_absolute():
        output_root = PROJECT_DIR / output_root
    output_root.mkdir(parents=True, exist_ok=True)
    results_csv = output_root / "hybrid_gpu_benchmark.csv"
    report_path = output_root / "hybrid_gpu_benchmark.md"
    selection_path = output_root / "selection.json"
    if (results_csv.is_file() and report_path.is_file() and selection_path.is_file()
            and not refresh_finalization):
        raise FileExistsError(f"Benchmark result already exists; refusing to overwrite {output_root}")
    payload_path = Path(experiments["initial_weights"])
    if not payload_path.is_absolute():
        payload_path = PROJECT_DIR / payload_path
    if not payload_path.is_file():
        raise FileNotFoundError(f"Run initialize_weights.py first: {payload_path}")
    from hybrid.common import sha256_file
    initial_hash = sha256_file(payload_path)
    payload = _load_shared_payload(payload_path)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    provenance = {
        "initial_checkpoint_sha256": initial_hash,
        "train_manifest_sha256": audit["train_manifest_sha256"],
    }
    device, _, _ = device_and_amp(True, "bfloat16")
    gpu_name = torch.cuda.get_device_name(device)
    total_memory = torch.cuda.get_device_properties(device).total_memory / (1024 ** 2)
    candidate_sizes = list(benchmark_config["image_budget_candidates"])
    checkpointing_count = int(benchmark_config["checkpointing_recheck_top_candidates"])
    modes = ("h0_supervised", "h1_pixel", "h2_latent")
    if results_csv.is_file():
        import pandas as pd
        rows = pd.read_csv(results_csv).to_dict(orient="records")
        for mode in modes:
            base_rows = [row for row in rows if row["mode"] == mode and
                         not bool(row["gradient_checkpointing"])]
            checkpoint_rows = [row for row in rows if row["mode"] == mode and
                               bool(row["gradient_checkpointing"])]
            if sorted(int(row["image_budget"]) for row in base_rows) != sorted(candidate_sizes):
                raise RuntimeError(f"Existing benchmark CSV lacks base candidates for {mode}")
            expected_checkpoint_rows = min(checkpointing_count,
                                           len([row for row in base_rows if row["status"] == "ok"]))
            if len(checkpoint_rows) != expected_checkpoint_rows:
                raise RuntimeError(f"Existing benchmark CSV lacks checkpointing rechecks for {mode}")
        for row in rows:
            row.update(provenance)
            row.setdefault("gpu_name", gpu_name)
            row.setdefault("gpu_total_memory_mib", total_memory)
        print(f"Resuming benchmark artifact finalization from {len(rows)} completed candidate records")
    else:
        rows = []
        for mode in modes:
            for budget in candidate_sizes:
                accumulation = max(1, (int(benchmark_config["target_effective_images"]) +
                                       int(budget) - 1) // int(budget))
                row = _candidate_result(mode, budget, accumulation, False, payload,
                                        payload_path, initial_hash,
                                        data_config, benchmark_config, output_root, provenance)
                row.update({"gpu_name": gpu_name, "gpu_total_memory_mib": total_memory})
                rows.append(row)
                print(f"{mode} budget={budget} accum={accumulation}: {row['status']} "
                      f"e2e_images_s={row.get('end_to_end_images_per_s', float('nan'))}")
                if row["status"] == "error":
                    raise RuntimeError(f"Benchmark failed for {mode}, image_budget={budget}: {row['error']}")

        for mode in modes:
            non_ckpt = [row for row in rows if row["mode"] == mode and
                        row["status"] == "ok" and not row["gradient_checkpointing"]]
            for prior in sorted(non_ckpt, key=lambda row: row["end_to_end_images_per_s"],
                                reverse=True)[:checkpointing_count]:
                row = _candidate_result(mode, prior["image_budget"], prior["grad_accumulation"],
                                        True, payload, payload_path, initial_hash,
                                        data_config, benchmark_config, output_root, provenance)
                row.update({"gpu_name": gpu_name, "gpu_total_memory_mib": total_memory})
                rows.append(row)
                print(f"{mode} checkpointing budget={row['image_budget']}: {row['status']} "
                      f"e2e_images_s={row.get('end_to_end_images_per_s', float('nan'))}")
                if row["status"] == "error":
                    raise RuntimeError(f"Checkpointing benchmark failed for {mode}: {row['error']}")

    selections = {}
    for mode in ("h0_supervised", "h1_pixel", "h2_latent"):
        selections[mode] = _choose(
            [row for row in rows if row["mode"] == mode],
            float(benchmark_config["oom_memory_margin"]),
            (int(benchmark_config["target_effective_images"])
             if mode in {"h1_pixel", "h2_latent"} else None),
            float(benchmark_config["checkpointing_throughput_tolerance"]),
        )
    train_path = PROJECT_DIR / experiments["train_config"]
    train_config = read_yaml(train_path)
    supervised_choice = selections["h0_supervised"]
    for key, value in {
        "image_budget": supervised_choice["image_budget"],
        "grad_accumulation": supervised_choice["grad_accumulation"],
        "num_workers": supervised_choice["num_workers"],
        "pin_memory": supervised_choice["pin_memory"],
        "optimizer_fused": supervised_choice["optimizer_fused"],
        "gradient_checkpointing": supervised_choice["gradient_checkpointing"],
    }.items():
        train_config[key] = value
    _write_yaml(train_path, train_config)

    for exp_key, selected_mode in (("h1_pretrain_config", "h1_pixel"),
                                   ("h2_pretrain_config", "h2_latent")):
        path = PROJECT_DIR / experiments[exp_key]
        config = read_yaml(path)
        selected = selections[selected_mode]
        for key in ("image_budget", "grad_accumulation", "num_workers", "pin_memory",
                    "optimizer_fused", "gradient_checkpointing"):
            config[key] = selected[key]
        _write_yaml(path, config)

    rows_sorted = sorted(rows, key=lambda row: (
        row["mode"], row["status"] != "ok", -row.get("end_to_end_images_per_s", 0)))
    write_csv(output_root / "hybrid_gpu_benchmark.csv", rows_sorted)
    report_lines = [
        "# Hybrid ConvNeXtV2–ViT GPU Training Benchmark", "",
        f"- GPU: {gpu_name}; total VRAM: {total_memory / 1024:.2f} GiB",
        f"- CUDA: {torch.version.cuda}; PyTorch: {torch.__version__}; timm: {timm.__version__}",
        "- Timed each configuration with the real data loader, H2D transfer, forward, loss, "
        "backward, gradient accumulation, clipping, and optimizer update.",
        "- Each configuration used 5 warmup updates and 20 measured updates; all workloads used "
        "a target effective batch of approximately 32 images.",
        "- GPU compute time uses CUDA events around forward/backward, clipping and optimizer step; "
        "it excludes DataLoader waiting and H2D transfer. End-to-end throughput includes both and "
        "is used for selection. Allocated/reserved metrics come from CUDA; total GPU utilization "
        "and used memory were sampled with nvidia-smi.",
        "- Gradient checkpointing was rechecked for the top two non-checkpointed throughput "
        "candidates per workload.", "",
    ]
    for mode, selected in selections.items():
        near_max = max(row["end_to_end_images_per_s"] for row in rows
                       if row["mode"] == mode and row["status"] == "ok")
        report_lines.extend([
            f"## {mode}", "",
            f"Selected: `image_budget={selected['image_budget']}`, "
            f"`grad_accumulation={selected['grad_accumulation']}`, "
            f"`gradient_checkpointing={str(selected['gradient_checkpointing']).lower()}`; "
            f"{selected['end_to_end_images_per_s']:.2f} end-to-end images/s, "
            f"{selected['gpu_compute_images_per_s']:.2f} compute images/s, "
            f"peak reserved {selected['peak_reserved_mib']:.0f} MiB "
            f"({selected['peak_reserved_fraction']:.1%} of VRAM), "
            f"{selected['update_p50_seconds']:.3f}s P50 / {selected['update_p95_seconds']:.3f}s P95 "
            f"per update, {selected['actual_images']} images / {selected['actual_studies']} Studies measured.",
            f"Highest measured rate was {near_max:.2f} images/s. The selection rule considers "
            f"configurations within {100*float(benchmark_config['checkpointing_throughput_tolerance']):.1f}% "
            "of the highest rate, then prefers more reserved-memory headroom.",
            "",
        ])
    report_lines.extend([
        "## Candidate table", "",
        "| Workload | Image budget | Grad accumulation | Effective batch target | Checkpointing | Status | Images / Studies / Subjects | Optimizer updates | End-to-end images/s | Compute images/s | P50/P95 update (s) | Peak alloc/reserved (MiB) | GPU util. | Mean total VRAM use | OOM/Error |",
        "|---|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ])
    for row in sorted(rows, key=lambda item: (item["mode"],
                                               -item.get("end_to_end_images_per_s", 0))):
        error_value = row.get("error", "")
        if not isinstance(error_value, str) or error_value.lower() == "nan":
            error_value = ""
        report_lines.append(
            f"| {row['mode']} | {row['image_budget']} | {row['grad_accumulation']} | "
            f"{row['target_effective_images']} | {row['gradient_checkpointing']} | {row['status']} | "
            f"{row.get('actual_images', 0)}/{row.get('actual_studies', 0)}/{row.get('actual_subjects', 0)} | "
            f"{row.get('optimizer_updates', 0)} | "
            f"{row.get('end_to_end_images_per_s', float('nan')):.2f} | "
            f"{row.get('gpu_compute_images_per_s', float('nan')):.2f} | "
            f"{row.get('update_p50_seconds', float('nan')):.3f}/{row.get('update_p95_seconds', float('nan')):.3f} | "
            f"{row.get('peak_allocated_mib', float('nan')):.0f}/{row.get('peak_reserved_mib', float('nan')):.0f} | "
            f"{row.get('gpu_utilization_mean_pct', float('nan')):.1f}% | "
            f"{row.get('gpu_memory_used_mean_fraction', float('nan')):.1%} | "
            f"{'OOM' if row.get('oom') else error_value} |"
        )
    (output_root / "hybrid_gpu_benchmark.md").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8")
    selection_path = output_root / "selection.json"
    write_json(selection_path, {"gpu": gpu_name, "selections": selections,
                                "benchmark_csv": str(output_root / "hybrid_gpu_benchmark.csv"),
                                "benchmark_report": str(output_root / "hybrid_gpu_benchmark.md")})
    print(f"Saved benchmark report: {output_root / 'hybrid_gpu_benchmark.md'}")
    print(f"Updated supervised and H1/H2 pretraining YAML with measured selections")
    return selections


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments-config", default="configs/hybrid_experiments.yaml")
    parser.add_argument("--refresh-finalization", action="store_true",
                        help="Rebuild the CSV/Markdown/selection from existing candidate records without rerunning measurements")
    args = parser.parse_args()
    run_benchmark(args.experiments_config, refresh_finalization=args.refresh_finalization)


if __name__ == "__main__":
    main()
