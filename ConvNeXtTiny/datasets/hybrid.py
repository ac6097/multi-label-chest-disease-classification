from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from .transforms import ResizePad


def build_hybrid_transform(size=320, train=False):
    """Return one shared, unnormalized [0, 1] image for both pretrained branches."""
    operations = [ResizePad(size), T.Grayscale(num_output_channels=3)]
    if train:
        operations.extend([
            T.RandomApply([T.ColorJitter(brightness=0.05, contrast=0.05)], p=0.5),
            T.RandomAffine(degrees=0, translate=(0.02, 0.02),
                           scale=(0.97, 1.03), fill=0),
        ])
    operations.append(T.ToTensor())
    return T.Compose(operations)


class OfficialTrainImageDataset(Dataset):
    """Unlabeled official train images; label columns are never read."""

    def __init__(self, csv_file, image_root, transform=None,
                 image_id_column="Dicom_id", subject_column="Subject_id",
                 study_column="Study_id"):
        self.csv_file = Path(csv_file)
        self.root = Path(image_root)
        self.transform = transform
        self.split = self.csv_file.stem
        self.image_id_column = image_id_column
        self.subject_column = subject_column
        self.study_column = study_column
        columns = [image_id_column, subject_column, study_column]
        frame = pd.read_csv(self.csv_file, usecols=columns, dtype={
            image_id_column: "string", subject_column: "string", study_column: "string",
        }, keep_default_na=True)
        if frame[columns].isna().any().any():
            raise ValueError(f"{self.csv_file} contains missing image or Study identifiers")
        if frame[image_id_column].duplicated().any():
            raise ValueError(f"{self.csv_file} contains duplicate {image_id_column} values")
        self.rows = frame.to_dict("records")

    def __len__(self):
        return len(self.rows)

    def image_path(self, index):
        row = self.rows[index]
        return (self.root / self.split / str(row[self.subject_column]) /
                str(row[self.study_column]) / f"{row[self.image_id_column]}.jpg")

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(self.image_path(index)) as source:
            image = source.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "image_id": str(row[self.image_id_column]),
            "subject_id": str(row[self.subject_column]),
            "study_id": str(row[self.study_column]),
        }
