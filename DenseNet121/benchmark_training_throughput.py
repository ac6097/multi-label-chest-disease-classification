"""Profile real Study batches through the DenseNet training path."""
from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import os
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from datasets import StudyBatchSampler, StudyDataset, build_transform, collate_studies
from engine.losses import MaskedBCEWithLogitsLoss
from model import ChestXrayClassifier
from utils.config import PROJECT_DIR, read_yaml, resolve_config_path, resolve_data_paths
from utils.seed import seed_everything


BASELINE_SHA256 = "8162986883188d168b013ad926f70eccddd559f61d9c922b05ab5db9eff7d477"
IMAGE_BUDGET_CASES = ((16, 2), (32, 1), (48, 1), (64, 1))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class TimedTransform:
    def __init__(self, transform):
        self.transform = transform
        self.seconds = 0.0

    def reset(self):
        self.seconds = 0.0

    def __call__(self, image):
        started = time.perf_counter()
        result = self.transform(image)
        self.seconds += time.perf_counter() - started
        return result


class TimedStudyDataset(Dataset):
    """Time image decode separately from the existing transform implementation."""
    def __init__(self, dataset: StudyDataset):
        self.dataset = dataset
        self.transform = TimedTransform(dataset.transform)
        self.dataset.transform = self.transform

    @property
    def image_counts(self):
        return self.dataset.image_counts

    @property
    def studies(self):
        return self.dataset.studies

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        self.transform.reset()
        started = time.perf_counter()
        item = self.dataset[index]
        elapsed = time.perf_counter() - started
        transform_seconds = self.transform.seconds
        item["_decode_convert_stack_seconds"] = max(elapsed - transform_seconds, 0.0)
        item["_transform_seconds"] = transform_seconds
        return item


def timed_collate(items):
    decode_seconds = [item.pop("_decode_convert_stack_seconds") for item in items]
    transform_seconds = [item.pop("_transform_seconds") for item in items]
    batch = collate_studies(items)
    batch["decode_convert_stack_seconds"] = torch.tensor(decode_seconds, dtype=torch.float64)
    batch["transform_seconds"] = torch.tensor(transform_seconds, dtype=torch.float64)
    return batch


class FILETIME(ctypes.Structure):
    _fields_ = [("dwLowDateTime", ctypes.c_uint32), ("dwHighDateTime", ctypes.c_uint32)]


def windows_cpu_counters():
    if os.name != "nt":
        return None
    idle, kernel, user = FILETIME(), FILETIME(), FILETIME()
    ok = ctypes.windll.kernel32.GetSystemTimes(
        ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user))
    if not ok:
        return None

    def value(item):
        return (item.dwHighDateTime << 32) | item.dwLowDateTime

    return value(idle), value(kernel), value(user)


class SystemSampler:
    """Sample whole-system CPU and NVIDIA GPU use without adding dependencies."""
    def __init__(self):
        self.lock = threading.Lock()
        self.active_case = None
        self.samples = {}
        self.stop_event = threading.Event()
        self.cpu_thread = None
        self.gpu_thread = None
        self.gpu_process = None

    def start(self):
        if os.name == "nt":
            self.cpu_thread = threading.Thread(target=self._sample_cpu, daemon=True)
            self.cpu_thread.start()
        try:
            self.gpu_process = subprocess.Popen(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                 "--format=csv,noheader,nounits", "--loop-ms=500"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
            self.gpu_thread = threading.Thread(target=self._sample_gpu, daemon=True)
            self.gpu_thread.start()
        except OSError:
            self.gpu_process = None

    def begin_case(self, name):
        with self.lock:
            self.active_case = name
            self.samples[name] = {"cpu_percent": [], "gpu_percent": [], "gpu_memory_mib": []}

    def end_case(self):
        with self.lock:
            self.active_case = None

    def _sample_cpu(self):
        previous = windows_cpu_counters()
        while not self.stop_event.wait(0.5):
            current = windows_cpu_counters()
            if previous and current:
                idle_delta = current[0] - previous[0]
                total_delta = (current[1] - previous[1]) + (current[2] - previous[2])
                if total_delta > 0:
                    value = max(0.0, min(100.0, 100.0 * (total_delta - idle_delta) / total_delta))
                    with self.lock:
                        if self.active_case is not None:
                            self.samples[self.active_case]["cpu_percent"].append(value)
            previous = current

    def _sample_gpu(self):
        process = self.gpu_process
        if process is None or process.stdout is None:
            return
        for line in process.stdout:
            try:
                gpu_percent, memory_mib = [float(value.strip()) for value in line.split(",")]
            except (TypeError, ValueError):
                continue
            with self.lock:
                if self.active_case is not None:
                    current = self.samples[self.active_case]
                    current["gpu_percent"].append(gpu_percent)
                    current["gpu_memory_mib"].append(memory_mib)

    def summary(self, name):
        with self.lock:
            samples = dict(self.samples.get(name, {}))
        result = {}
        for metric, values in samples.items():
            result[metric] = {
                "mean": float(np.mean(values)) if values else None,
                "p50": float(np.percentile(values, 50)) if values else None,
                "p95": float(np.percentile(values, 95)) if values else None,
                "samples": len(values),
            }
        return result

    def close(self):
        self.stop_event.set()
        if self.gpu_process is not None:
            self.gpu_process.terminate()
            try:
                self.gpu_process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.gpu_process.kill()
        for thread in (self.cpu_thread, self.gpu_thread):
            if thread is not None:
                thread.join(timeout=2)


def event_summary(event_pairs):
    values = [start.elapsed_time(end) for start, end in event_pairs]
    if not values:
        return {"mean_ms": None, "p50_ms": None, "p95_ms": None}
    return {
        "mean_ms": float(np.mean(values)),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
    }


def timed_batch(loader_iter):
    started = time.perf_counter()
    batch = next(loader_iter)
    return batch, time.perf_counter() - started


def run_case(data_cfg, checkpoint_state, device, amp_dtype, image_budget, accumulation,
             loader_case, target_images, warmup_batches, system_sampler, case_name):
    name, workers, persistent, prefetch = loader_case
    seed_everything(42)
    base_dataset = StudyDataset(
        data_cfg["train_csv"], data_cfg["image_root"],
        transform=build_transform(320, train=True),
        label_column=data_cfg["label_column"], label_semantics=data_cfg["label_semantics"],
        image_id_column=data_cfg["image_id_column"],
        subject_column=data_cfg["subject_column"], study_column=data_cfg["study_column"])
    dataset = TimedStudyDataset(base_dataset)
    batch_sampler = StudyBatchSampler(dataset, image_budget, shuffle=True, seed=42)
    planned_batches = batch_sampler._batches()
    study_is_valid = [bool(item["valid"].any()) for item in dataset.studies]
    window_valid_counts = []
    for start in range(0, len(planned_batches), accumulation):
        window = planned_batches[start:start + accumulation]
        window_valid_counts.append(sum(
            study_is_valid[index] for batch_indices in window for index in batch_indices))
    loader_args = {
        "batch_sampler": batch_sampler,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": timed_collate,
    }
    if workers:
        loader_args["persistent_workers"] = persistent
        loader_args["prefetch_factor"] = prefetch
    loader = DataLoader(dataset, **loader_args)

    model_config = dict(checkpoint_state["config"]["model_config"])
    model_config["pretrained"] = False
    model = ChestXrayClassifier(**{key: value for key, value in model_config.items()
                                   if key != "name"}).to(device)
    model.load_state_dict(checkpoint_state["model"], strict=True)
    model.train()
    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": 2e-5},
        {"params": model.vector_classifier.parameters(), "lr": 2e-4},
    ], weight_decay=1e-4)
    criterion = MaskedBCEWithLogitsLoss().to(device)
    optimizer.zero_grad(set_to_none=True)
    amp_enabled = device.type == "cuda"

    batch_iterator = None
    event_pairs = {key: [] for key in
                   ("transfer", "forward", "loss", "backward", "optimizer_step")}
    measured_decode, measured_transform = [], []
    loader_waits, host_sync_seconds = [], []
    batch_images, batch_studies = [], []
    measured_images = measured_studies = measured_batches = measured_updates = 0
    startup_seconds = None
    measured_started = None
    wall_seconds = None
    peak_allocated_gib = peak_reserved_gib = None
    system_sampler.begin_case(case_name)
    try:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        startup_started = time.perf_counter()
        batch_iterator = iter(loader)
        first_batch, first_wait = timed_batch(batch_iterator)
        startup_seconds = time.perf_counter() - startup_started
        pending = [(first_batch, first_wait)]
        batch_index = 0
        while batch_index < len(loader):
            if batch_index == warmup_batches and measured_started is None:
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                    torch.cuda.reset_peak_memory_stats(device)
                measured_started = time.perf_counter()
            if pending:
                batch, wait_seconds = pending.pop()
            else:
                batch, wait_seconds = timed_batch(batch_iterator)

            is_measured = batch_index >= warmup_batches
            if is_measured:
                loader_waits.append(wait_seconds)
                measured_decode.extend(batch["decode_convert_stack_seconds"].tolist())
                measured_transform.extend(batch["transform_seconds"].tolist())
                batch_images.append(int(batch["images"].shape[0]))
                batch_studies.append(int(batch["num_studies"]))

            transfer_start = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
            transfer_end = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
            if transfer_start:
                transfer_start.record()
            images = batch["images"].to(device, non_blocking=True)
            study_index = batch["study_index"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            known = batch["known"].to(device, non_blocking=True)
            if transfer_end:
                transfer_end.record()
                if is_measured:
                    event_pairs["transfer"].append((transfer_start, transfer_end))

            forward_start = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
            forward_end = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
            if forward_start:
                forward_start.record()
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                logits = model(images, study_index, batch["num_studies"])
            logits = logits.float()
            if forward_end:
                forward_end.record()
                if is_measured:
                    event_pairs["forward"].append((forward_start, forward_end))

            loss_start = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
            loss_end = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
            if loss_start:
                loss_start.record()
            loss = criterion(logits, labels, known)
            if loss_end:
                loss_end.record()
                if is_measured:
                    event_pairs["loss"].append((loss_start, loss_end))

            sync_started = time.perf_counter()
            eligible_studies = int(known.any(dim=1).sum().item())
            if is_measured:
                host_sync_seconds.append(time.perf_counter() - sync_started)
            window_valid = max(window_valid_counts[batch_index // accumulation], 1)
            scaled_loss = loss * (eligible_studies / window_valid)
            backward_start = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
            backward_end = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
            if backward_start:
                backward_start.record()
            scaled_loss.backward()
            if backward_end:
                backward_end.record()
                if is_measured:
                    event_pairs["backward"].append((backward_start, backward_end))

            sync_started = time.perf_counter()
            _ = float(loss.detach()) * eligible_studies
            if is_measured:
                host_sync_seconds.append(time.perf_counter() - sync_started)

            update_now = ((batch_index + 1) % accumulation == 0 or
                          batch_index + 1 == len(loader))
            if update_now:
                optimizer_start = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
                optimizer_end = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
                if optimizer_start:
                    optimizer_start.record()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                sync_started = time.perf_counter()
                _ = float(norm)
                if is_measured:
                    host_sync_seconds.append(time.perf_counter() - sync_started)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if optimizer_end:
                    optimizer_end.record()
                    if is_measured:
                        event_pairs["optimizer_step"].append((optimizer_start, optimizer_end))
                if is_measured:
                    measured_updates += 1

            if is_measured:
                measured_images += int(batch["images"].shape[0])
                measured_studies += int(batch["num_studies"])
                measured_batches += 1
            batch_index += 1
            if (is_measured and measured_images >= target_images and
                    (batch_index % accumulation == 0 or batch_index == len(loader))):
                break

        if device.type == "cuda":
            torch.cuda.synchronize(device)
            wall_seconds = time.perf_counter() - measured_started
            peak_allocated_gib = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            peak_reserved_gib = torch.cuda.max_memory_reserved(device) / (1024 ** 3)
            peak_inactive_split_gib = (
                torch.cuda.memory_stats(device).get("inactive_split_bytes.all.peak", 0) / (1024 ** 3))
        else:
            wall_seconds = time.perf_counter() - measured_started
            peak_inactive_split_gib = None
        system_sampler.end_case()
        metrics = {stage: event_summary(pairs) for stage, pairs in event_pairs.items()}
        row = {
            "case": case_name, "status": "ok", "image_budget": image_budget,
            "grad_accumulation": accumulation, "num_workers": workers,
            "persistent_workers": bool(persistent), "prefetch_factor": prefetch if workers else None,
            "loader_startup_seconds": startup_seconds,
            "measured_batches": measured_batches, "measured_images": measured_images,
            "measured_studies": measured_studies, "optimizer_updates_measured": measured_updates,
            "updates_per_1000_images": (1000.0 * measured_updates / measured_images),
            "updates_per_epoch": int(np.ceil(len(planned_batches) / accumulation)),
            "planned_batches_per_epoch": len(planned_batches),
            "mean_images_per_batch": float(np.mean(batch_images)),
            "std_images_per_batch": float(np.std(batch_images)),
            "p50_images_per_batch": float(np.percentile(batch_images, 50)),
            "p95_images_per_batch": float(np.percentile(batch_images, 95)),
            "min_images_per_batch": int(np.min(batch_images)),
            "max_images_per_batch": int(np.max(batch_images)),
            "mean_studies_per_batch": float(np.mean(batch_studies)),
            "std_studies_per_batch": float(np.std(batch_studies)),
            "p50_studies_per_batch": float(np.percentile(batch_studies, 50)),
            "p95_studies_per_batch": float(np.percentile(batch_studies, 95)),
            "min_studies_per_batch": int(np.min(batch_studies)),
            "max_studies_per_batch": int(np.max(batch_studies)),
            "wall_seconds": wall_seconds,
            "images_per_second": measured_images / wall_seconds,
            "studies_per_second": measured_studies / wall_seconds,
            "mean_batch_wall_ms": 1000.0 * wall_seconds / max(measured_batches, 1),
            "mean_optimizer_update_wall_ms": 1000.0 * wall_seconds / max(measured_updates, 1),
            "mean_loader_wait_ms": 1000.0 * float(np.mean(loader_waits)),
            "decode_convert_stack_ms_per_image": 1000.0 * float(np.sum(measured_decode)) / measured_images,
            "resize_augment_normalize_ms_per_image": 1000.0 * float(np.sum(measured_transform)) / measured_images,
            "explicit_host_sync_ms_per_batch": 1000.0 * float(np.sum(host_sync_seconds)) / measured_batches,
            "peak_allocated_gib": peak_allocated_gib,
            "peak_reserved_gib": peak_reserved_gib,
            "peak_inactive_split_gib": peak_inactive_split_gib,
            "stages": metrics,
            "system_utilization": system_sampler.summary(case_name),
        }
        return row
    finally:
        system_sampler.end_case()
        if batch_iterator is not None and hasattr(batch_iterator, "_shutdown_workers"):
            batch_iterator._shutdown_workers()
        del batch_iterator, loader, model, optimizer, criterion, dataset, base_dataset, batch_sampler
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def run_case_safe(data_cfg, checkpoint_state, device, amp_dtype, image_budget, accumulation,
                  loader_case, target_images, warmup_batches, system_sampler, case_name):
    try:
        return run_case(data_cfg, checkpoint_state, device, amp_dtype, image_budget, accumulation,
                        loader_case, target_images, warmup_batches, system_sampler, case_name)
    except torch.cuda.OutOfMemoryError as exc:
        system_sampler.end_case()
        allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 3)
        inactive_split = torch.cuda.memory_stats(device).get(
            "inactive_split_bytes.all.peak", 0) / (1024 ** 3)
        gc.collect()
        torch.cuda.empty_cache()
        return {
            "case": case_name, "status": "oom", "image_budget": image_budget,
            "grad_accumulation": accumulation, "num_workers": loader_case[1],
            "persistent_workers": bool(loader_case[2]),
            "prefetch_factor": loader_case[3] if loader_case[1] else None,
            "error": str(exc), "peak_allocated_gib": allocated, "peak_reserved_gib": reserved,
            "peak_inactive_split_gib": inactive_split,
            "system_utilization": system_sampler.summary(case_name),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-config", default=str(PROJECT_DIR / "configs/data.yaml"))
    parser.add_argument("--checkpoint", default=str(PROJECT_DIR / "checkpoint/best/best_by_macro_auc.pth"))
    parser.add_argument("--output-dir", default=str(PROJECT_DIR / "outputs/experiments/densenet_gpu_benchmark"))
    parser.add_argument("--images-per-case", type=int, default=1024)
    parser.add_argument("--warmup-batches", type=int, default=4)
    args = parser.parse_args()
    if args.images_per_case < 1 or args.warmup_batches < 1:
        parser.error("--images-per-case and --warmup-batches must both be positive")

    data_path = resolve_config_path(args.data_config)
    data_cfg = resolve_data_paths(read_yaml(data_path), data_path)
    checkpoint_path = resolve_config_path(args.checkpoint)
    checkpoint_digest = sha256(checkpoint_path)
    if checkpoint_digest.lower() != BASELINE_SHA256:
        raise ValueError(f"Baseline checkpoint SHA-256 mismatch: {checkpoint_digest}")
    checkpoint_state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = dict(checkpoint_state["config"]["model_config"])
    if (int(checkpoint_state.get("epoch", 0)) != 6 or model_config.get("head") != "gap" or
            model_config.get("study_fusion") != "logit_mean" or model_config.get("cbam", False)):
        raise ValueError("Throughput benchmark requires the original Epoch 6 standard GAP checkpoint")
    if int(data_cfg["image_size"]) != 320:
        raise ValueError("Throughput benchmark is defined for 320x320 inputs")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the requested GPU throughput benchmark")
    device = torch.device("cuda")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The benchmark requires BF16 support to match the training configuration")
    amp_dtype = torch.bfloat16
    gpu_name = torch.cuda.get_device_name(device)
    output_dir = resolve_config_path(args.output_dir)
    report_path = output_dir / "benchmark_report.json"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing benchmark artifacts: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    current_workers = int(data_cfg["num_workers"])
    current_persistent = bool(data_cfg.get("persistent_workers", False))
    current_prefetch = int(data_cfg.get("prefetch_factor", 2))
    worker_cases = [
        ("current", current_workers, current_persistent, current_prefetch),
        ("workers_0", 0, False, 0),
        ("workers_2_persistent", 2, True, 2),
        ("workers_4_persistent", 4, True, 2),
        ("workers_8_persistent", 8, True, 2),
        ("workers_4_prefetch_4", 4, True, 4),
    ]
    unique_cases = []
    seen = set()
    for case in worker_cases:
        key = case[1:]
        if key not in seen:
            seen.add(key)
            unique_cases.append(case)

    sampler = SystemSampler()
    sampler.start()
    rows = []
    try:
        worker_rows = []
        for case in unique_cases:
            row = run_case_safe(
                data_cfg, checkpoint_state, device, amp_dtype, 16, 2, case,
                max(args.images_per_case, 1), args.warmup_batches, sampler,
                f"loader_{case[0]}")
            worker_rows.append(row)
            rows.append(row)

        valid_worker_rows = [row for row in worker_rows if row["status"] == "ok"]
        if not valid_worker_rows:
            raise RuntimeError("Every DataLoader profile failed; no batch recommendation can be made")
        best_loader = max(valid_worker_rows, key=lambda row: row["images_per_second"])
        selected_loader = (
            f"selected_{best_loader['num_workers']}_workers_"
            f"persistent_{str(best_loader['persistent_workers']).lower()}_"
            f"prefetch_{best_loader['prefetch_factor'] or 'default'}")
        selected_tuple = (
            selected_loader, best_loader["num_workers"],
            best_loader["persistent_workers"], best_loader["prefetch_factor"] or 2)
        for budget, accumulation in IMAGE_BUDGET_CASES:
            row = run_case_safe(
                data_cfg, checkpoint_state, device, amp_dtype, budget, accumulation,
                selected_tuple, max(args.images_per_case, 1), args.warmup_batches,
                sampler, f"batch_{budget}_{accumulation}")
            rows.append(row)
    finally:
        sampler.close()

    successful = [row for row in rows if row["status"] == "ok"]
    batch_rows = [row for row in successful if row["case"].startswith("batch_")]
    if not batch_rows:
        batch_rows = [row for row in successful if row["case"].startswith("loader_")]
    best_batch = max(batch_rows, key=lambda row: row["images_per_second"])
    train_dataset = StudyDataset(
        data_cfg["train_csv"], data_cfg["image_root"], transform=None,
        label_column=data_cfg["label_column"], label_semantics=data_cfg["label_semantics"],
        image_id_column=data_cfg["image_id_column"],
        subject_column=data_cfg["subject_column"], study_column=data_cfg["study_column"])
    summary = {
        "device": gpu_name,
        "device_memory_bytes": int(torch.cuda.get_device_properties(device).total_memory),
        "cuda_version": torch.version.cuda,
        "pytorch_version": torch.__version__,
        "amp_dtype": str(amp_dtype),
        "dataset": {
            "train_csv": str(data_cfg["train_csv"]),
            "studies": len(train_dataset),
            "images": int(sum(train_dataset.image_counts)),
            "image_size": 320,
            "data_loader_cases": [row for row in rows if row["case"].startswith("loader_")],
            "batch_cases": [row for row in rows if row["case"].startswith("batch_")],
            "selected_data_loader": {
                "num_workers": best_loader["num_workers"],
                "persistent_workers": best_loader["persistent_workers"],
                "prefetch_factor": best_loader["prefetch_factor"],
                "images_per_second": best_loader["images_per_second"],
                "studies_per_second": best_loader["studies_per_second"],
            },
            "highest_throughput_batch": {
                "image_budget": best_batch["image_budget"],
                "grad_accumulation": best_batch["grad_accumulation"],
                "images_per_second": best_batch["images_per_second"],
                "studies_per_second": best_batch["studies_per_second"],
            },
            "baseline_checkpoint_sha256": checkpoint_digest,
            "results": rows,
        },
    }
    report_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    flat_rows = []
    for row in rows:
        flat = {key: value for key, value in row.items()
                if key not in {"stages", "system_utilization"}}
        for stage, timings in row.get("stages", {}).items():
            for metric, value in timings.items():
                flat[f"{stage}_{metric}"] = value
        for metric, stats in row.get("system_utilization", {}).items():
            flat[f"{metric}_mean"] = stats["mean"]
            flat[f"{metric}_p50"] = stats["p50"]
            flat[f"{metric}_p95"] = stats["p95"]
        flat_rows.append(flat)
    pd.DataFrame(flat_rows).to_csv(output_dir / "benchmark_summary.csv", index=False)
    print(json.dumps({
        "output_dir": str(output_dir),
        "selected_data_loader": summary["dataset"]["selected_data_loader"],
        "highest_throughput_batch": summary["dataset"]["highest_throughput_batch"],
        "cases": [{"case": row["case"], "status": row["status"],
                   "images_per_second": row.get("images_per_second"),
                   "studies_per_second": row.get("studies_per_second"),
                   "peak_allocated_gib": row.get("peak_allocated_gib"),
                   "peak_reserved_gib": row.get("peak_reserved_gib")}
                  for row in rows],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
