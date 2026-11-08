"""Official Train/Validation data access and deterministic Study batching."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler


WORK_DIR = Path(__file__).resolve().parent
PROJECT_DIR = WORK_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from datasets.dataset import StudyDataset, collate_studies  # noqa: E402
from datasets.transforms import build_transform  # noqa: E402
from hybrid.common import seed_worker  # noqa: E402


class EpochTaggedStudyDataset(Dataset):
    """Make augmentation deterministic per (seed, epoch, Study) across resumes."""

    def __init__(self, dataset: StudyDataset, seed: int):
        self.dataset = dataset
        self.seed = int(seed)

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, name):
        if name == "dataset":
            raise AttributeError(name)
        return getattr(self.dataset, name)

    def __getitem__(self, key):
        if isinstance(key, (tuple, list)):
            index, epoch = key
        else:
            index, epoch = key, 0
        index, epoch = int(index), int(epoch)
        seed = (self.seed + 1_000_003 * epoch + 9_176 * index) % (2 ** 32)
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state()
        try:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            return self.dataset[index]
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.set_rng_state(torch_state)


class EpochStudyBatchSampler(Sampler):
    """Budget whole Studies without dropping or repeating any Study."""

    def __init__(self, dataset, image_budget: int, shuffle: bool, seed: int):
        self.dataset = dataset
        self.image_budget = max(1, int(image_budget))
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def _batches(self):
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.Random(self.seed + self.epoch).shuffle(indices)
        batches, current, image_count = [], [], 0
        for index in indices:
            count = int(self.dataset.image_counts[index])
            if current and image_count + count > self.image_budget:
                batches.append(current)
                current, image_count = [], 0
            current.append(index)
            image_count += count
            if image_count >= self.image_budget:
                batches.append(current)
                current, image_count = [], 0
        if current:
            batches.append(current)
        flattened = [index for batch in batches for index in batch]
        if len(flattened) != len(self.dataset) or set(flattened) != set(range(len(self.dataset))):
            raise RuntimeError("Study sampler must visit each Study exactly once")
        return batches

    def __iter__(self):
        for batch in self._batches():
            yield [(index, self.epoch) for index in batch]

    def __len__(self):
        return len(self._batches())


def read_data_config(config_path: Path, image_size: int | None = None):
    import yaml

    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    for key in ("image_root", "train_csv", "val_csv"):
        config[key] = (PROJECT_DIR / config[key]).resolve()
    if not config["train_csv"].is_file() or not config["val_csv"].is_file():
        raise FileNotFoundError("The configured official Train/Validation CSV is missing")
    if not config["image_root"].is_dir():
        raise FileNotFoundError(f"Configured image root does not exist: {config['image_root']}")
    resolved_image_size = int(config.get("image_size", 0) if image_size is None else image_size)
    if resolved_image_size < 32 or resolved_image_size % 32:
        raise ValueError("DINOv3 image_size must be a positive multiple of 32")
    config["image_size"] = resolved_image_size
    if config.get("label_semantics") != "closed_world":
        raise ValueError("This route requires the confirmed official closed-world label semantics")
    return config


def build_datasets(config_path: Path, seed: int, image_size: int | None = None):
    config = read_data_config(config_path, image_size=image_size)
    resolved_image_size = int(config["image_size"])
    train_base = StudyDataset(
        config["train_csv"], config["image_root"],
        transform=build_transform(resolved_image_size, train=True),
        label_column=config["label_column"],
        label_semantics=config["label_semantics"],
        image_id_column=config["image_id_column"],
        subject_column=config["subject_column"],
        study_column=config["study_column"],
    )
    valid = StudyDataset(
        config["val_csv"], config["image_root"],
        transform=build_transform(resolved_image_size, train=False),
        label_column=config["label_column"],
        label_semantics=config["label_semantics"],
        image_id_column=config["image_id_column"],
        subject_column=config["subject_column"],
        study_column=config["study_column"],
    )
    if train_base.split != "train" or valid.split != "valid":
        raise ValueError("Only official train and valid splits are permitted in this experiment")
    return EpochTaggedStudyDataset(train_base, seed), EpochTaggedStudyDataset(valid, seed), config


def make_loader(dataset, image_budget: int, shuffle: bool, seed: int,
                num_workers: int, pin_memory: bool, prefetch_factor: int = 2,
                collate_fn=collate_studies):
    sampler = EpochStudyBatchSampler(dataset, image_budget, shuffle, seed)
    generator = torch.Generator().manual_seed(int(seed))
    kwargs = {
        "dataset": dataset,
        "batch_sampler": sampler,
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
        "collate_fn": collate_fn,
        "generator": generator,
        "worker_init_fn": seed_worker if num_workers else None,
        "persistent_workers": bool(num_workers),
    }
    if num_workers:
        kwargs["prefetch_factor"] = int(prefetch_factor)
    return DataLoader(**kwargs), sampler, generator


def shutdown_loader(loader):
    if loader is None:
        return
    iterator = getattr(loader, "_iterator", None)
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if shutdown is not None:
        shutdown()
