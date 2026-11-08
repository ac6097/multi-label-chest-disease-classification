import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd

from datasets.dataset import parse_positive_ids
from hybrid.common import PROJECT_DIR, resolve_data_config, sha256_file, write_csv, write_json


def _audit_split(split_name, csv_path, image_root, config, check_labels):
    columns = [config["image_id_column"], config["subject_column"], config["study_column"]]
    if check_labels:
        columns.append(config["label_column"])
    frame = pd.read_csv(csv_path, usecols=columns, dtype={
        config["image_id_column"]: "string",
        config["subject_column"]: "string",
        config["study_column"]: "string",
    }, keep_default_na=True)
    id_columns = columns[:3]
    if frame[id_columns].isna().any().any():
        raise ValueError(f"{split_name} CSV contains missing image or Study identifiers")
    if frame[config["image_id_column"]].duplicated().any():
        raise ValueError(f"{split_name} CSV contains duplicate image IDs")

    study_labels = defaultdict(set)
    if check_labels:
        for _, row in frame.iterrows():
            label = row.get(config["label_column"])
            if pd.notna(label) and str(label).strip():
                study_labels[(str(row[config["subject_column"]]),
                              str(row[config["study_column"]]))].add(parse_positive_ids(label))
        conflict_count = sum(len(labels) > 1 for labels in study_labels.values())
        blank_rows = int((frame[config["label_column"]].fillna("").astype(str).str.strip() == "").sum())
        if conflict_count:
            raise ValueError(f"{split_name} has conflicting labels within {conflict_count} Studies")
        if blank_rows:
            raise ValueError(f"{split_name} has {blank_rows} rows without classification labels")

    manifest = []
    missing = []
    subject_col = config["subject_column"]
    study_col = config["study_column"]
    image_col = config["image_id_column"]
    for _, row in frame.iterrows():
        subject_id, study_id, image_id = (str(row[subject_col]), str(row[study_col]),
                                          str(row[image_col]))
        relative_path = Path(split_name) / subject_id / study_id / f"{image_id}.jpg"
        full_path = Path(image_root) / relative_path
        manifest.append({"split": split_name, "subject_id": subject_id,
                         "study_id": study_id, "image_id": image_id,
                         "relative_path": relative_path.as_posix()})
        if not full_path.is_file():
            missing.append(relative_path.as_posix())
    if missing:
        raise FileNotFoundError(f"{split_name} is missing {len(missing)} manifest image(s); first={missing[0]}")
    return {
        "manifest": manifest,
        "image_rows": len(frame),
        "studies": len({(item["subject_id"], item["study_id"]) for item in manifest}),
        "subjects": len({item["subject_id"] for item in manifest}),
        "missing_images": len(missing),
        "source_csv_sha256": sha256_file(csv_path),
    }


def audit_and_write(data_config_path="configs/hybrid_data.yaml", output_dir=None):
    config = resolve_data_config(data_config_path)
    for key in ("image_root", "train_csv", "val_csv"):
        if not Path(config[key]).exists():
            raise FileNotFoundError(f"Configured {key} does not exist: {config[key]}")
    if config.get("label_semantics") != "closed_world":
        raise ValueError("Hybrid classification requires the confirmed closed_world label semantics")

    train = _audit_split("train", config["train_csv"], config["image_root"], config, True)
    valid = _audit_split("valid", config["val_csv"], config["image_root"], config, True)
    train_subjects = {item["subject_id"] for item in train["manifest"]}
    valid_subjects = {item["subject_id"] for item in valid["manifest"]}
    overlap = sorted(train_subjects & valid_subjects)
    if overlap:
        raise ValueError(f"Train/Validation Subject overlap detected: {len(overlap)}")

    output_dir = Path(output_dir or PROJECT_DIR / "outputs/hybrid/shared/manifests")
    if not output_dir.is_absolute():
        output_dir = PROJECT_DIR / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, manifest_rows in (("train_images.csv", train["manifest"]),
                                    ("valid_images.csv", valid["manifest"])):
        manifest_path = output_dir / filename
        if manifest_path.exists():
            existing = pd.read_csv(manifest_path, dtype="string", keep_default_na=False)
            expected = pd.DataFrame(manifest_rows, columns=[
                "split", "subject_id", "study_id", "image_id", "relative_path"]
            ).astype("string")
            if not existing.equals(expected):
                raise FileExistsError(f"Existing image manifest differs from the current official split: {manifest_path}")
        else:
            write_csv(manifest_path, manifest_rows)
    train_manifest_hash = sha256_file(output_dir / "train_images.csv")
    valid_manifest_hash = sha256_file(output_dir / "valid_images.csv")
    report = {
        "image_root": config["image_root"],
        "train_csv": config["train_csv"],
        "valid_csv": config["val_csv"],
        "test_split_read": False,
        "label_semantics": config["label_semantics"],
        "image_size": config["image_size"],
        "train": {key: value for key, value in train.items() if key != "manifest"},
        "valid": {key: value for key, value in valid.items() if key != "manifest"},
        "train_valid_subject_overlap": 0,
        "train_manifest_sha256": train_manifest_hash,
        "valid_manifest_sha256": valid_manifest_hash,
        "manifest_files": {
            "train": str(output_dir / "train_images.csv"),
            "valid": str(output_dir / "valid_images.csv"),
        },
        "usage": {
            "H0": {"supervised_train_manifest": "train_images.csv", "validation_manifest": "valid_images.csv"},
            "H1": {"self_supervised_manifest": "train_images.csv", "supervised_train_manifest": "train_images.csv", "validation_manifest": "valid_images.csv"},
            "H2": {"self_supervised_manifest": "train_images.csv", "supervised_train_manifest": "train_images.csv", "validation_manifest": "valid_images.csv"},
        },
    }
    report_path = output_dir / "data_audit.json"
    if report_path.exists():
        existing_report = json.loads(report_path.read_text(encoding="utf-8"))
        for key in ("train_manifest_sha256", "valid_manifest_sha256", "train_valid_subject_overlap"):
            if existing_report.get(key) != report.get(key):
                raise FileExistsError(f"Existing data audit differs from the current official split: {report_path}")
        for split in ("train", "valid"):
            if existing_report.get(split, {}).get("source_csv_sha256") != report[split]["source_csv_sha256"]:
                raise FileExistsError(f"{split} CSV content changed since the saved audit: {report_path}")
    else:
        write_json(report_path, report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", default="configs/hybrid_data.yaml")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    report = audit_and_write(args.data_config, args.output_dir)
    print("Official Train/Validation audit passed:")
    print(f"  Train: {report['train']['image_rows']} images / {report['train']['studies']} Studies / "
          f"{report['train']['subjects']} Subjects")
    print(f"  Valid: {report['valid']['image_rows']} images / {report['valid']['studies']} Studies / "
          f"{report['valid']['subjects']} Subjects")
    print("  Missing images: 0; Subject overlap: 0; Test CSV read: no")
    print(f"  Manifests: {report['manifest_files']}")


if __name__ == "__main__":
    main()
