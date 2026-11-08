from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


CLASS_IDS = tuple(range(1, 11))


def parse_positive_ids(value):
    """Parse the observed semicolon-separated positive class IDs without guessing negatives."""
    if pd.isna(value) or str(value).strip() == "":
        return None
    tokens = [token.strip() for token in str(value).split(";") if token.strip()]
    try:
        values = [int(token) for token in tokens]
    except ValueError as exc:
        raise ValueError(f"Unrecognized Predict_class value: {value!r}") from exc
    if not values or any(class_id not in CLASS_IDS for class_id in values):
        raise ValueError(f"Predict_class IDs must be in 1..10: {value!r}")
    if len(values) != len(set(values)):
        raise ValueError(f"Predict_class contains duplicate IDs: {value!r}")
    return tuple(sorted(values))


class StudyDataset(Dataset):
    """One item per (Subject_id, Study_id), with all of that Study's images."""

    def __init__(self, csv_file, image_root, transform=None, label_column="Predict_class",
                 label_semantics="unresolved", image_id_column="Dicom_id",
                 subject_column="Subject_id", study_column="Study_id"):
        self.csv_file = Path(csv_file)
        self.root = Path(image_root)
        self.transform = transform
        self.label_column = label_column
        self.image_id_column = image_id_column
        self.subject_column = subject_column
        self.study_column = study_column
        if label_semantics not in {"unresolved", "positive_only", "closed_world"}:
            raise ValueError("label_semantics must be unresolved, positive_only, or closed_world")
        self.label_semantics = label_semantics
        dtype = {image_id_column: "string", subject_column: "string", study_column: "string"}
        frame = pd.read_csv(self.csv_file, dtype=dtype, keep_default_na=True)
        required = {image_id_column, subject_column, study_column}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{self.csv_file} is missing required columns: {missing}")
        if frame[list(required)].isna().any().any():
            raise ValueError(f"{self.csv_file} contains missing image or Study identifiers")
        if frame[image_id_column].duplicated().any():
            raise ValueError(f"{self.csv_file} contains duplicate {image_id_column} values")

        self.split = self.csv_file.stem
        self.studies = []
        group_columns = [subject_column, study_column]
        for (subject_id, study_id), group in frame.groupby(group_columns, sort=False, dropna=False):
            raw_values = []
            if label_column in group.columns:
                for value in group[label_column].tolist():
                    parsed = parse_positive_ids(value)
                    if parsed is not None:
                        raw_values.append(parsed)
            if len(set(raw_values)) > 1:
                raise ValueError(f"Conflicting labels within Study {(subject_id, study_id)}")
            positive_ids = raw_values[0] if raw_values else None
            target, valid = self._encode_labels(positive_ids)
            image_paths = [
                self.root / self.split / str(subject_id) / str(study_id) / f"{image_id}.jpg"
                for image_id in group[image_id_column].astype(str).tolist()
            ]
            self.studies.append({
                "subject_id": str(subject_id),
                "study_id": str(study_id),
                "image_ids": group[image_id_column].astype(str).tolist(),
                "image_paths": image_paths,
                "target": target,
                "valid": valid,
            })
        self._image_counts = [len(study["image_paths"]) for study in self.studies]
        self._study_keys = [(study["subject_id"], study["study_id"]) for study in self.studies]

    def _encode_labels(self, positive_ids):
        target = torch.zeros(len(CLASS_IDS), dtype=torch.float32)
        valid = torch.zeros(len(CLASS_IDS), dtype=torch.bool)
        if positive_ids is None:
            return target, valid
        for class_id in positive_ids:
            target[class_id - 1] = 1.0
            valid[class_id - 1] = True
        if self.label_semantics == "closed_world":
            # This interpretation must only be selected after the competition owner confirms it.
            valid[:] = True
        # For unresolved/positive-only semantics, omitted IDs remain masked unknowns.
        return target, valid

    @property
    def image_counts(self):
        return self._image_counts

    @property
    def study_keys(self):
        return self._study_keys

    def label_tensors(self):
        if not self.studies:
            return torch.empty((0, 10)), torch.empty((0, 10), dtype=torch.bool)
        return (torch.stack([item["target"] for item in self.studies]),
                torch.stack([item["valid"] for item in self.studies]))

    def __len__(self):
        return len(self.studies)

    def __getitem__(self, index):
        study = self.studies[index]
        images = []
        for path in study["image_paths"]:
            with Image.open(path) as source:
                image = source.convert("RGB")
            if self.transform:
                image = self.transform(image)
            images.append(image)
        return {
            "images": torch.stack(images),
            "labels": study["target"],
            "known": study["valid"],
            "subject_id": study["subject_id"],
            "study_id": study["study_id"],
            "image_ids": study["image_ids"],
        }


def collate_studies(items):
    """Flatten a variable number of images and retain an explicit image→Study map."""
    if not items:
        raise ValueError("Cannot collate an empty Study batch")
    images = [item["images"] for item in items]
    counts = torch.tensor([image.shape[0] for image in images], dtype=torch.long)
    study_index = torch.repeat_interleave(torch.arange(len(items)), counts)
    return {
        "images": torch.cat(images, dim=0),
        "study_index": study_index,
        "num_studies": len(items),
        "image_counts": counts,
        "labels": torch.stack([item["labels"] for item in items]),
        "known": torch.stack([item["known"] for item in items]),
        "subject_id": [item["subject_id"] for item in items],
        "study_id": [item["study_id"] for item in items],
        "image_ids": [item["image_ids"] for item in items],
    }
