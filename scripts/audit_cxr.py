#!/usr/bin/env python
"""Read-only audit of cxr_first; writes reproducible reports beneath docs/."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import time
from io import BytesIO
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError


CLASSES = [
    "Enlarged Cardiomediastinum", "Pneumothorax", "Consolidation", "Pneumonia",
    "Edema", "Cardiomegaly", "Atelectasis", "Lung Opacity", "Pleural Effusion",
    "No Finding",
]
SPLITS = ["train", "valid", "test"]
ID_COLS = ["Subject_id", "Study_id"]
VIEW_FRONTAL = {"AP", "PA"}
VIEW_LATERAL = {"LATERAL", "LL"}
# User-confirmed project rule for train/valid: omitted IDs in a non-empty
# Predict_class list are negatives. Blank or malformed labels remain errors.
TRAIN_VALID_LABEL_SEMANTICS = "closed_world"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def parse_codes(value: str) -> tuple[list[int], str | None]:
    """Parse the observed semicolon-delimited code list without inferring negatives."""
    value = str(value).strip()
    if not value:
        return [], "blank"
    tokens = value.split(";")
    if any(not re.fullmatch(r"\d+", token.strip()) for token in tokens):
        return [], "malformed"
    codes = [int(token) for token in tokens]
    if len(codes) != len(set(codes)):
        return codes, "duplicate_code"
    if any(code < 1 or code > 10 for code in codes):
        return codes, "out_of_range"
    return codes, None


def fmt_num(value, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "未识别"
    if isinstance(value, (float, np.floating)):
        return f"{value:.{digits}f}"
    return str(value)


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def markdown_table(frame: pd.DataFrame, floatfmt: str | None = None, index: bool = False) -> str:
    """Small dependency-free Markdown table formatter."""
    table = frame.copy()
    if index:
        table = table.reset_index()
    columns = [str(column) for column in table.columns]

    def cell(value) -> str:
        if pd.isna(value):
            return "NA"
        if isinstance(value, (float, np.floating)) and floatfmt:
            value = format(float(value), floatfmt)
        return str(value).replace("|", "\\|").replace("\n", " ")

    rows = [[cell(value) for value in row] for row in table.itertuples(index=False, name=None)]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def collect_inventory(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    seen = 0
    for directory, _, names in os.walk(root):
        for name in names:
            path = Path(directory) / name
            records.append({
                "relative_path": path.relative_to(root).as_posix(),
                "extension": path.suffix.lower() or "[none]",
                "bytes": path.stat().st_size,
            })
            seen += 1
            if seen % 10000 == 0:
                print(f"file inventory: {seen} files", flush=True)
    files = pd.DataFrame(records).sort_values("relative_path").reset_index(drop=True)
    types = files.groupby("extension", as_index=False).agg(file_count=("bytes", "size"), total_bytes=("bytes", "sum"))
    return files, types.sort_values("extension")


def image_path(root: Path, split: str, row: pd.Series) -> Path:
    # Observed hierarchy is split/Subject_id/Study_id/Dicom_id.jpg.
    return root / split / str(row["Subject_id"]) / str(row["Study_id"]) / f"{row['Dicom_id']}.jpg"


def classify_view(value: str) -> str:
    view = str(value).strip().upper()
    if not view:
        return "缺失"
    if view in VIEW_FRONTAL:
        return "正位"
    if view in VIEW_LATERAL:
        return "侧位"
    return "其他/复合"


def summarize_subject_study(frame: pd.DataFrame, split: str) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    frame = frame.copy()
    pair = frame.groupby(ID_COLS, dropna=False, sort=False)
    study_sizes = pair.size().rename("image_count").reset_index()
    subject_studies = frame.groupby("Subject_id", dropna=False)["Study_id"].nunique().rename("study_count").reset_index()
    summary = {
        "split": split,
        "subject_count": int(frame["Subject_id"].nunique()),
        "study_count_composite_key": int(len(study_sizes)),
        "image_rows": int(len(frame)),
        "images_per_study_mean": float(study_sizes.image_count.mean()),
        "images_per_study_median": float(study_sizes.image_count.median()),
        "images_per_study_max": int(study_sizes.image_count.max()),
        "single_image_studies": int((study_sizes.image_count == 1).sum()),
        "two_image_studies": int((study_sizes.image_count == 2).sum()),
        "three_plus_image_studies": int((study_sizes.image_count >= 3).sum()),
        "subjects_with_study_count_mean": float(subject_studies.study_count.mean()),
        "subjects_with_study_count_median": float(subject_studies.study_count.median()),
        "subjects_with_study_count_max": int(subject_studies.study_count.max()),
    }
    return summary, study_sizes, subject_studies


def inventory_images(root: Path, files: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame, list[str]]:
    records = []
    errors = []
    digest_paths: dict[str, list[str]] = defaultdict(list)
    image_files = [root / rel for rel in files.loc[files.extension.isin([".jpg", ".jpeg", ".png"]), "relative_path"]]
    start = time.time()
    for i, path in enumerate(image_files, start=1):
        rel = path.relative_to(root).as_posix()
        try:
            image_bytes = path.read_bytes()
            file_hash = hashlib.sha256(image_bytes).hexdigest()
            digest_paths[file_hash].append(rel)
            with Image.open(BytesIO(image_bytes)) as image:
                image.load()
                image_format = image.format or "unknown"
                mode = image.mode
                width, height = image.size
            records.append({
                "relative_path": rel, "format": image_format, "mode": mode,
                "width": width, "height": height, "aspect_ratio_width_over_height": width / height,
                "channel_type": "灰度模式" if mode in {"1", "L", "I", "I;16", "F"} else "彩色/调色板/带透明通道",
                "sha256": file_hash,
                "bytes": path.stat().st_size,
            })
        except (OSError, ValueError, UnidentifiedImageError, SyntaxError) as exc:
            errors.append(f"{rel}: {type(exc).__name__}: {exc}")
        if i % 5000 == 0:
            print(f"image scan: {i}/{len(image_files)}", flush=True)

    metadata = pd.DataFrame(records)
    exact_groups = [paths for paths in digest_paths.values() if len(paths) > 1]
    duplicate_rows = []
    for group_no, paths in enumerate(exact_groups, start=1):
        duplicate_rows.append({"duplicate_type": "文件字节 SHA-256", "group_id": group_no, "count": len(paths), "paths": " | ".join(paths)})
    summary = {
        "image_paths_seen": len(image_files),
        "readable_images": int(len(metadata)),
        "unreadable_images": len(errors),
        "format_counts": metadata["format"].value_counts().to_dict() if len(metadata) else {},
        "mode_counts": metadata["mode"].value_counts().to_dict() if len(metadata) else {},
        "channel_type_counts": metadata["channel_type"].value_counts().to_dict() if len(metadata) else {},
        "unique_dimension_count": int(metadata.groupby(["width", "height"]).ngroups) if len(metadata) else 0,
        "width_min_median_max": [int(metadata.width.min()), float(metadata.width.median()), int(metadata.width.max())] if len(metadata) else [],
        "height_min_median_max": [int(metadata.height.min()), float(metadata.height.median()), int(metadata.height.max())] if len(metadata) else [],
        "aspect_ratio_min_median_max": [float(metadata.aspect_ratio_width_over_height.min()), float(metadata.aspect_ratio_width_over_height.median()), float(metadata.aspect_ratio_width_over_height.max())] if len(metadata) else [],
        "exact_file_duplicate_groups": len(exact_groups),
        "decoded_pixel_duplicate_groups": None,
        "scan_seconds": round(time.time() - start, 2),
    }
    return summary, metadata, pd.DataFrame(duplicate_rows, columns=["duplicate_type", "group_id", "count", "paths"]), errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("cxr_first"))
    parser.add_argument("--output-root", type=Path, default=Path("docs"))
    parser.add_argument("--reuse-image-metadata", action="store_true", help="Reuse docs/tables/image_metadata.csv from a prior full scan")
    args = parser.parse_args()
    root = args.data_root.resolve()
    out = args.output_root.resolve()
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    files, file_types = collect_inventory(root)
    save_csv(files, out / "tables" / "source_file_inventory.csv")
    save_csv(file_types, out / "tables" / "source_file_types.csv")

    frames: dict[str, pd.DataFrame] = {}
    split_summaries, size_tables, subject_tables = [], {}, {}
    code_errors, row_paths, labels_by_split = [], {}, {}
    for split in SPLITS:
        csv_path = root / f"{split}.csv"
        # Disable NA inference and preserve identifiers such as 00001 as strings.
        frame = pd.read_csv(csv_path, dtype=str, keep_default_na=False, na_filter=False)
        frames[split] = frame
        expected = {"Dicom_id", "Subject_id", "Study_id", "ViewPosition", "PatientOrientation", "Rows", "Columns"}
        if split != "test":
            expected.add("Predict_class")
        missing = sorted(expected - set(frame.columns))
        if missing:
            raise ValueError(f"{csv_path} missing expected fields: {missing}")
        summary, study_sizes, subject_studies = summarize_subject_study(frame, split)
        split_summaries.append(summary)
        size_tables[split] = study_sizes
        subject_tables[split] = subject_studies

        if "Predict_class" in frame:
            parsed = frame["Predict_class"].map(parse_codes)
            bad = parsed.map(lambda x: x[1])
            if split in {"train", "valid"} and bad.notna().any():
                raise ValueError(
                    f"{csv_path} has blank or malformed Predict_class values; "
                    "closed-world encoding only applies to valid, non-empty positive-ID lists."
                )
            if bad.notna().any():
                for index in frame.index[bad.notna()]:
                    code_errors.append({"split": split, "row": int(index) + 2, "Dicom_id": frame.at[index, "Dicom_id"], "raw_value": frame.at[index, "Predict_class"], "issue": bad.at[index]})
            out_of_range = parsed.map(lambda item: any(code not in range(1, 11) for code in item[0]))
            if out_of_range.any():
                raise ValueError(f"Out-of-range class IDs in {csv_path}")

        row_paths[split] = [image_path(root, split, row) for _, row in frame.iterrows()]

    split_summary_df = pd.DataFrame(split_summaries)
    save_csv(split_summary_df, out / "tables" / "split_counts.csv")
    for split, table in size_tables.items():
        table.insert(0, "split", split)
        save_csv(table, out / "tables" / f"study_image_counts_{split}.csv")
        save_csv(subject_tables[split].assign(split=split), out / "tables" / f"subject_study_counts_{split}.csv")
    study_size_distribution = (
        pd.concat([size_tables[s] for s in SPLITS], ignore_index=True)
        .groupby(["split", "image_count"], as_index=False).size()
        .rename(columns={"size": "study_count"})
    )
    subject_size_distribution = (
        pd.concat([subject_tables[s].assign(split=s) for s in SPLITS], ignore_index=True)
        .groupby(["split", "study_count"], as_index=False).size()
        .rename(columns={"size": "subject_count"})
    )
    save_csv(study_size_distribution, out / "tables" / "study_image_count_distribution.csv")
    save_csv(subject_size_distribution, out / "tables" / "subject_study_count_distribution.csv")

    # Composite identity is authoritative throughout this script.
    pair_map: dict[tuple[str, str], list[str]] = {}
    study_to_subjects: dict[str, set[str]] = defaultdict(set)
    all_dicom = []
    path_lookup = {}
    for split, frame in frames.items():
        for (_, row), path in zip(frame.iterrows(), row_paths[split]):
            key = (row.Subject_id, row.Study_id)
            pair_map.setdefault(key, []).append(split)
            study_to_subjects[row.Study_id].add(row.Subject_id)
            all_dicom.append((split, row.Dicom_id))
            path_lookup[(split, row.Dicom_id)] = path
    duplicate_study_id = {k: sorted(v) for k, v in study_to_subjects.items() if len(v) > 1}
    global_study_ids = set(study_to_subjects)
    global_pairs = set(pair_map)
    split_subjects = {s: set(f.Subject_id) for s, f in frames.items()}
    overlap_rows = []
    for left, right in [("train", "valid"), ("train", "test"), ("valid", "test")]:
        overlap = sorted(split_subjects[left] & split_subjects[right])
        for subject in overlap:
            overlap_rows.append({"split_a": left, "split_b": right, "Subject_id": subject})
    save_csv(pd.DataFrame(overlap_rows, columns=["split_a", "split_b", "Subject_id"]), out / "tables" / "subject_split_overlap.csv")

    # Match every CSV image row to its expected file path and scan all source images.
    if args.reuse_image_metadata:
        image_meta = pd.read_csv(out / "tables" / "image_metadata.csv", dtype={"relative_path": str, "format": str, "mode": str, "channel_type": str, "sha256": str})
        duplicates = pd.read_csv(out / "tables" / "image_duplicate_groups.csv", dtype=str, keep_default_na=False)
        error_file = out / "tables" / "image_read_errors.txt"
        image_errors = error_file.read_text(encoding="utf-8").splitlines() if error_file.exists() else []
        image_summary = {
            "image_paths_seen": int(files.extension.isin([".jpg", ".jpeg", ".png"]).sum()),
            "readable_images": int(len(image_meta)), "unreadable_images": len(image_errors),
            "format_counts": image_meta["format"].value_counts().to_dict(),
            "mode_counts": image_meta["mode"].value_counts().to_dict(),
            "channel_type_counts": image_meta["channel_type"].value_counts().to_dict(),
            "unique_dimension_count": int(image_meta.groupby(["width", "height"]).ngroups),
            "width_min_median_max": [int(image_meta.width.min()), float(image_meta.width.median()), int(image_meta.width.max())],
            "height_min_median_max": [int(image_meta.height.min()), float(image_meta.height.median()), int(image_meta.height.max())],
            "aspect_ratio_min_median_max": [float(image_meta.aspect_ratio_width_over_height.min()), float(image_meta.aspect_ratio_width_over_height.median()), float(image_meta.aspect_ratio_width_over_height.max())],
            "exact_file_duplicate_groups": int((duplicates.get("duplicate_type", pd.Series(dtype=str)) == "文件字节 SHA-256").sum()),
            "decoded_pixel_duplicate_groups": None, "scan_seconds": "reused cache",
        }
    else:
        image_summary, image_meta, duplicates, image_errors = inventory_images(root, files)
    save_csv(image_meta, out / "tables" / "image_metadata.csv")
    save_csv(duplicates, out / "tables" / "image_duplicate_groups.csv")
    with (out / "tables" / "image_read_errors.txt").open("w", encoding="utf-8") as stream:
        stream.write("\n".join(image_errors))
    (out / "tables" / "image_summary.json").write_text(json.dumps(image_summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    indexed_paths = set(image_meta.relative_path) if len(image_meta) else set()
    duplicate_group_count = int(len(duplicates))
    duplicate_image_count = int(pd.to_numeric(duplicates["count"], errors="coerce").fillna(0).sum()) if duplicate_group_count else 0
    duplicate_cross_split_groups = 0
    duplicate_cross_study_groups = 0
    if duplicate_group_count:
        for value in duplicates["paths"]:
            paths = [item.split("/") for item in str(value).split(" | ")]
            if len({parts[0] for parts in paths}) > 1:
                duplicate_cross_split_groups += 1
            if len({tuple(parts[:3]) for parts in paths}) > 1:
                duplicate_cross_study_groups += 1
    ref_rows = []
    for split, frame in frames.items():
        for row, path in zip(frame.to_dict("records"), row_paths[split]):
            ref_rows.append({"split": split, "Subject_id": row["Subject_id"], "Study_id": row["Study_id"], "Dicom_id": row["Dicom_id"], "relative_path": path.relative_to(root).as_posix(), "exists": path.is_file()})
    refs = pd.DataFrame(ref_rows)
    save_csv(refs.loc[~refs.exists], out / "tables" / "missing_image_references.csv")
    reference_paths = set(refs.relative_path)
    source_image_paths = set(files.loc[files.extension.isin([".jpg", ".jpeg", ".png"]), "relative_path"])
    unreferenced = sorted(source_image_paths - reference_paths)
    save_csv(pd.DataFrame({"relative_path": unreferenced}), out / "tables" / "unreferenced_images.csv")
    duplicate_dicoms = pd.DataFrame([
        {"split": split, "Dicom_id": dicom_id, "occurrences": count}
        for (split, dicom_id), count in Counter(all_dicom).items() if count > 1
    ])
    save_csv(pd.DataFrame(duplicate_dicoms, columns=["split", "Dicom_id", "occurrences"]), out / "tables" / "duplicate_csv_image_ids.csv")

    # Add image metadata to source rows; path shape is based on observed hierarchy.
    meta_by_path = image_meta.set_index("relative_path").to_dict("index") if len(image_meta) else {}
    view_rows = []
    dim_mismatches = []
    for split, frame in frames.items():
        for row, path in zip(frame.to_dict("records"), row_paths[split]):
            meta = meta_by_path.get(path.relative_to(root).as_posix())
            if meta:
                view_rows.append({"split": split, "Subject_id": row["Subject_id"], "Study_id": row["Study_id"], "ViewPosition": row["ViewPosition"], "view_class": classify_view(row["ViewPosition"])})
                try:
                    if int(row["Rows"]) != meta["height"] or int(row["Columns"]) != meta["width"]:
                        dim_mismatches.append({"split": split, "Dicom_id": row["Dicom_id"], "csv_Rows": row["Rows"], "csv_Columns": row["Columns"], "image_height": meta["height"], "image_width": meta["width"]})
                except (ValueError, TypeError):
                    dim_mismatches.append({"split": split, "Dicom_id": row["Dicom_id"], "csv_Rows": row["Rows"], "csv_Columns": row["Columns"], "image_height": meta["height"], "image_width": meta["width"]})
    save_csv(pd.DataFrame(dim_mismatches, columns=["split", "Dicom_id", "csv_Rows", "csv_Columns", "image_height", "image_width"]), out / "tables" / "dimension_metadata_mismatches.csv")
    for split, frame in frames.items():
        for col in ["ViewPosition", "PatientOrientation"]:
            dist = frame[col].replace("", "[空值]").value_counts(dropna=False).rename_axis("value").reset_index(name="image_count")
            dist.insert(0, "split", split)
            dist.insert(1, "field", col)
            if split == "train" and col == "ViewPosition":
                metadata_distributions = dist.copy()
            else:
                metadata_distributions = pd.concat([metadata_distributions, dist], ignore_index=True)
    save_csv(metadata_distributions, out / "tables" / "metadata_value_counts.csv")

    view_df = pd.DataFrame(view_rows)
    view_combo_rows = []
    for split in SPLITS:
        sf = view_df[view_df.split == split]
        for (subject, study), group in sf.groupby(ID_COLS, sort=False):
            counts = Counter(group.view_class)
            combo = "+".join(sorted(counts))
            view_combo_rows.append({"split": split, "Subject_id": subject, "Study_id": study, "image_count": len(group), "view_combination": combo, "view_labels": "+".join(sorted(set(group.ViewPosition.replace("", "缺失"))))})
    view_combos = pd.DataFrame(view_combo_rows)
    combo_counts = view_combos.groupby(["split", "view_combination"], as_index=False).size().rename(columns={"size": "study_count"})
    save_csv(combo_counts, out / "tables" / "study_view_combinations.csv")

    # Study-level class-code analysis. Multiple images of a Study are collapsed once.
    study_label_rows = []
    label_conflicts = []
    for split in ["train", "valid"]:
        frame = frames[split]
        for (subject, study), group in frame.groupby(ID_COLS, sort=False):
            raw_sets = group.Predict_class.astype(str).unique().tolist()
            code_sets = [tuple(sorted(parse_codes(value)[0])) for value in raw_sets]
            if len(set(code_sets)) > 1:
                label_conflicts.append({"split": split, "Subject_id": subject, "Study_id": study, "raw_values": " | ".join(raw_sets)})
            # Conflicting image-level annotations are left unresolved; never pick the first row.
            codes = set(code_sets[0]) if len(set(code_sets)) == 1 else set()
            study_label_rows.append({"split": split, "Subject_id": subject, "Study_id": study, "raw_codes": ";".join(map(str, sorted(codes))), **{f"class_{c}": int(c in codes) for c in range(1, 11)}, "positive_code_count": len(codes)})
    study_labels = pd.DataFrame(study_label_rows)
    if label_conflicts:
        raise ValueError("Conflicting image-level labels exist within a Study; resolve them before closed-world analysis.")
    save_csv(study_labels, out / "tables" / "study_labels_deduplicated.csv")
    save_csv(pd.DataFrame(label_conflicts, columns=["split", "Subject_id", "Study_id", "raw_values"]), out / "tables" / "study_label_conflicts.csv")
    save_csv(pd.DataFrame(code_errors, columns=["split", "row", "Dicom_id", "raw_value", "issue"]), out / "tables" / "label_code_parse_issues.csv")

    label_stats = []
    conditional_weights = []
    cooc_frames, conditional_frames = [], []
    for split in ["train", "valid"]:
        sf = study_labels[study_labels.split == split]
        n_studies = len(sf)
        y = sf[[f"class_{i}" for i in range(1, 11)]].to_numpy(dtype=int)
        counts = y.sum(axis=0)
        cooc = y.T @ y
        rate = np.divide(cooc, counts[:, None], out=np.zeros_like(cooc, dtype=float), where=counts[:, None] != 0)
        for i, c in enumerate(range(1, 11)):
            npos = int(counts[i])
            unresolved = n_studies - npos
            ratio = unresolved / npos if npos else np.inf
            label_stats.append({
                "split": split, "class_id": c, "category": CLASSES[c - 1], "study_count": n_studies,
                "N_positive_explicitly_listed": npos, "N_negative_explicitly_encoded": 0,
                "N_unknown_or_unresolved_not_listed": 0,
                "N_valid_binary_explicit": n_studies,
                "N_negative_under_confirmed_rule": unresolved,
                "N_unknown": 0,
                "N_valid": n_studies,
                "positive_share_among_explicit_binary_labels": npos / n_studies if n_studies else np.nan,
                "negative_share_among_explicit_binary_labels": unresolved / n_studies if n_studies else np.nan,
                "positive_code_share_of_all_studies": npos / n_studies if n_studies else np.nan,
                "negative_positive_ratio": ratio,
                "closed_world_candidate_N_negative_if_absence_means_negative": unresolved,
                "closed_world_candidate_positive_rate_if_absence_means_negative": npos / n_studies if n_studies else np.nan,
                "closed_world_candidate_neg_pos_ratio_if_absence_means_negative": unresolved / npos if npos else np.inf,
            })
            if split == "train":
                conditional_weights.append({
                    "class_id": c, "category": CLASSES[c - 1], "N_pos_listed_studies": npos,
                    "N_not_listed_studies": unresolved, "A_BCE_pos_weight_1": 1.0,
                    "B_raw_neg_pos_ratio_actual": ratio,
                    "C_sqrt_ratio_actual": np.sqrt(ratio),
                    "D_cap5_actual": min(ratio, 5), "D_cap10_actual": min(ratio, 10),
                    "D_cap20_actual": min(ratio, 20),
                    "B_raw_closed_world_hypothesis_only": ratio,
                    "C_sqrt_closed_world_hypothesis_only": np.sqrt(ratio),
                    "D_cap5_closed_world_hypothesis_only": min(ratio, 5),
                    "D_cap10_closed_world_hypothesis_only": min(ratio, 10),
                    "D_cap20_closed_world_hypothesis_only": min(ratio, 20),
                    "status": "用户已确认训练集未列出类别为阴性" if npos else "N_pos=0；该类权重不可估计",
                })
        for i, a in enumerate(range(1, 11)):
            for j, b in enumerate(range(1, 11)):
                cooc_frames.append({"split": split, "class_i": a, "class_i_name": CLASSES[a - 1], "class_j": b, "class_j_name": CLASSES[b - 1], "cooccurrence_study_count": int(cooc[i, j])})
                conditional_frames.append({"split": split, "condition_class_i": a, "condition_class_i_name": CLASSES[a - 1], "class_j": b, "class_j_name": CLASSES[b - 1], "P_j_listed_given_i_listed": rate[i, j] if counts[i] else np.nan, "denominator_i_listed_studies": int(counts[i])})
    label_stats_df = pd.DataFrame(label_stats)
    save_csv(label_stats_df, out / "tables" / "label_distribution_study_level.csv")
    label_rate_pivot = label_stats_df.pivot(index=["class_id", "category"], columns="split", values="positive_code_share_of_all_studies").reset_index()
    label_count_pivot = label_stats_df.pivot(index=["class_id", "category"], columns="split", values="N_positive_explicitly_listed").reset_index()
    comparison = label_rate_pivot.merge(label_count_pivot, on=["class_id", "category"], suffixes=("_listed_code_share", "_listed_studies"))
    comparison["valid_minus_train_listed_code_share"] = comparison["valid_listed_code_share"] - comparison["train_listed_code_share"]
    save_csv(comparison, out / "tables" / "train_valid_label_code_rate_comparison.csv")
    save_csv(pd.DataFrame(conditional_weights), out / "tables" / "bce_weight_candidates.csv")
    save_csv(pd.DataFrame(cooc_frames), out / "tables" / "label_cooccurrence.csv")
    save_csv(pd.DataFrame(conditional_frames), out / "tables" / "label_conditional_cooccurrence.csv")

    positive_hist = study_labels.groupby("split").positive_code_count.value_counts().rename("study_count").reset_index()
    save_csv(positive_hist, out / "tables" / "study_positive_label_count_distribution.csv")
    no_finding_rows = []
    for split in ["train", "valid"]:
        sf = study_labels[study_labels.split == split]
        disease_cols = [f"class_{i}" for i in range(1, 10)]
        nf = sf.class_10 == 1
        any_disease = sf[disease_cols].any(axis=1)
            no_finding_rows.append({"split": split, "study_count": len(sf), "no_finding_listed": int(nf.sum()), "no_finding_and_any_disease_listed": int((nf & any_disease).sum()), "no_finding_only_listed": int((nf & ~any_disease).sum()), "neither_no_finding_nor_disease_listed": int((~nf & ~any_disease).sum()), "note": "共现数按显式阳性ID统计；train/valid未列出类别按用户确认规则为阴性；No Finding与病种共现是否矛盾仍待定义"})
    no_finding = pd.DataFrame(no_finding_rows)
    save_csv(no_finding, out / "tables" / "no_finding_relationship.csv")

    # Figures: descriptive of positive code mentions only; no inferred negatives.
    train_stats = label_stats_df[label_stats_df.split == "train"].copy()
    fig, ax = plt.subplots(figsize=(11, 5.8))
    bars = ax.barh(train_stats.category, train_stats.N_positive_explicitly_listed, color="#3274a1")
    ax.invert_yaxis(); ax.set_xlabel("Studies with class ID explicitly listed"); ax.set_title("Train set: Study-level listed class-code counts")
    for bar in bars: ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2, f" {int(bar.get_width()):,}", va="center", fontsize=8)
    fig.tight_layout(); fig.savefig(out / "figures" / "train_study_class_code_counts.png", dpi=160); plt.close(fig)

    train_y = study_labels[study_labels.split == "train"][[f"class_{i}" for i in range(1, 11)]].to_numpy(dtype=int)
    train_cooc = train_y.T @ train_y
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(train_cooc, cmap="Blues")
    ax.set_xticks(range(10), [f"{i+1}. {x}" for i, x in enumerate(CLASSES)], rotation=55, ha="right", fontsize=8)
    ax.set_yticks(range(10), [f"{i+1}. {x}" for i, x in enumerate(CLASSES)], fontsize=8)
    for i in range(10):
        for j in range(10): ax.text(j, i, str(int(train_cooc[i, j])), ha="center", va="center", fontsize=7, color="black")
    ax.set_title("Train set: Study-level listed-code co-occurrence")
    fig.colorbar(im, ax=ax, label="Study count"); fig.tight_layout(); fig.savefig(out / "figures" / "train_label_cooccurrence.png", dpi=160); plt.close(fig)

    study_count_table = pd.concat([size_tables[s] for s in SPLITS], ignore_index=True)
    counts_by_size = study_count_table.groupby(["split", "image_count"]).size().unstack(0, fill_value=0)
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    for ax, split, color in zip(axes, SPLITS, ["#3274a1", "#e1812c", "#3a923a"]):
        values = counts_by_size[split] if split in counts_by_size else pd.Series(dtype=int)
        ax.bar(values.index, values.values, color=color)
        ax.set_title(split)
        ax.set_ylabel("Study count")
    axes[-1].set_xlabel("Images per Study")
    fig.suptitle("Study image-count distribution by split")
    fig.tight_layout()
    fig.savefig(out / "figures" / "study_image_count_distribution.png", dpi=160); plt.close(fig)

    # An audit manifest captures the evidence and makes outputs reproducible.
    csv_hashes = {f"{s}.csv": sha256_file(root / f"{s}.csv") for s in SPLITS}
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": str(root), "output_root": str(out), "source_csv_sha256": csv_hashes,
        "source_file_count": int(len(files)), "source_file_type_counts": file_types.set_index("extension").file_count.to_dict(),
        "split_counts": split_summaries, "image_audit": image_summary,
        "study_id_is_globally_unique_across_subjects": not bool(duplicate_study_id),
        "study_ids_used_by_multiple_subjects": duplicate_study_id,
        "study_ids_global_count": len(global_study_ids), "subject_study_pair_global_count": len(global_pairs),
        "subjects_shared_between_splits": overlap_rows,
        "csv_image_path_missing_count": int((~refs.exists).sum()), "unreferenced_image_count": len(unreferenced),
        "duplicate_csv_image_id_count": int(len(duplicate_dicoms)), "image_reference_count": int(len(refs)),
        "exact_duplicate_group_count": duplicate_group_count, "exact_duplicate_image_count": duplicate_image_count,
        "exact_duplicate_cross_split_group_count": duplicate_cross_split_groups,
        "exact_duplicate_cross_study_group_count": duplicate_cross_study_groups,
        "metadata_fields": {s: list(frames[s].columns) for s in SPLITS},
        "view_field_value_counts": {s: frames[s].ViewPosition.value_counts(dropna=False).to_dict() for s in SPLITS},
        "patient_orientation_value_counts": {s: frames[s].PatientOrientation.value_counts(dropna=False).to_dict() for s in SPLITS},
        "label_encoding": "Predict_class is a semicolon-delimited list of integer class IDs; no explicit negative/unknown state found in source files.",
        "label_to_name_mapping_source": "User-provided ordered class list mapped to official IDs 1-10; no mapping legend found in cxr_first.",
        "training_validation_label_semantics": f"{TRAIN_VALID_LABEL_SEMANTICS}: user confirmed omitted IDs in valid train/valid Predict_class lists are negative; test labels are not used.",
    }
    (out / "tables" / "audit_manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    # Detailed markdown reports.
    split_md = markdown_table(split_summary_df, floatfmt=".2f")
    fmt_counts = "; ".join(f"{r.extension}: {int(r.file_count):,} 个 ({int(r.total_bytes):,} bytes)" for r in file_types.itertuples())
    max_rows = []
    for split in SPLITS:
        max_count = int(size_tables[split].image_count.max())
        max_rows.extend([{**row, "split": split} for row in size_tables[split].loc[size_tables[split].image_count == max_count].to_dict("records")])
    max_df = pd.DataFrame(max_rows)
    max_md = markdown_table(max_df) if len(max_df) else "无"
    image_modes_md = markdown_table(image_meta["mode"].value_counts().to_frame("images"), index=True) if len(image_meta) else "无"
    common_sizes_md = markdown_table(image_meta.groupby(["width", "height"], as_index=False).size().sort_values("size", ascending=False).head(15)) if len(image_meta) else "无"
    metadata_md = markdown_table(metadata_distributions)
    combo_md = markdown_table(combo_counts)
    subject_distribution_md = markdown_table(subject_size_distribution)
    docs = out
    (docs / "dataset_audit.md").write_text(f"""# 数据集结构与完整性审计

## 范围与可复现性

审计脚本：`scripts/audit_cxr.py`。命令：`python scripts/audit_cxr.py --data-root cxr_first --output-root docs`。脚本以只读方式访问源目录，统计产物写入 `docs/`。源 CSV SHA-256、详细文件清单和运行摘要见 `tables/audit_manifest.json`、`tables/source_file_inventory.csv`。本次递归扫描发现 {len(files):,} 个源文件：{fmt_counts}。

| 划分 | Subject 数 | Study 数（Subject_id + Study_id） | Image 行数 | 每 Study 图片均值 | 中位数 | 单图 Study | 双图 Study | ≥3 图 Study | 最大图片数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(f"| {r.split} | {int(r.subject_count):,} | {int(r.study_count_composite_key):,} | {int(r.image_rows):,} | {r.images_per_study_mean:.3f} | {r.images_per_study_median:.0f} | {int(r.single_image_studies):,} | {int(r.two_image_studies):,} | {int(r.three_plus_image_studies):,} | {int(r.images_per_study_max)} |" for r in split_summary_df.itertuples())}

`Study_id` 全局唯一（跨 Subject）={not bool(duplicate_study_id)}；跨划分 Subject 重叠数={len(overlap_rows)}。以联合键作为统计和校验主键。Subject 每人 Study 数分布详见 `tables/subject_study_counts_*.csv`；Study 图片数明细详见 `tables/study_image_counts_*.csv`。各划分最大图片数及对应检查：

{max_md}

### Subject 每人 Study 数分布

{subject_distribution_md}

![Study 图片数分布](figures/study_image_count_distribution.png)

## 图片和完整性

CSV 的 `Dicom_id`、`Subject_id`、`Study_id` 与文件目录层级匹配为 `split/Subject_id/Study_id/Dicom_id.jpg`。CSV 图像引用 {len(refs):,} 行，缺失引用 {(~refs.exists).sum():,}；目录中未被 CSV 引用的 JPEG/PNG {len(unreferenced):,}；重复 Dicom_id 行 {len(duplicate_dicoms):,}。损坏/无法读取图像 {len(image_errors):,}。文件字节 SHA-256 完全重复组 {duplicate_group_count} 组、涉及 {duplicate_image_count} 张，跨 Study 重复组 {duplicate_cross_study_groups}，跨划分重复组 {duplicate_cross_split_groups}。本次不做不同编码下的感知重复检测。重复明细、缺图、孤立图和维度不匹配见 `tables/`。

- 可读图像：{image_summary['readable_images']:,}；格式：{image_summary['format_counts']}。
- Pillow 模式分布：

{image_modes_md}

- 宽度（最小/中位/最大）：{image_summary['width_min_median_max']}；高度：{image_summary['height_min_median_max']}；宽高比范围和中位数：{image_summary['aspect_ratio_min_median_max']}。
- 最常见的实际像素尺寸：

{common_sizes_md}

CSV 的 `Rows` / `Columns` 与图片解码尺寸不一致行数：{len(dim_mismatches)}。统计像素通道模式，不对灰度图重新编码。完整图像明细见 `tables/image_metadata.csv`。

## Study 图片数和视角组合

分类规则：AP/PA 归正位，LATERAL/LL 归侧位，PA LLD 等未完全映射取值归“其他/复合”，空值保留为缺失。正侧位组合按 Study 计数：

{combo_md}

| 字段 | 训练集空值 | 验证集空值 | 测试集空值 |
|---|---:|---:|---:|
| ViewPosition | {int((frames['train'].ViewPosition=='').sum())} | {int((frames['valid'].ViewPosition=='').sum())} | {int((frames['test'].ViewPosition=='').sum())} |
| PatientOrientation | {int((frames['train'].PatientOrientation=='').sum())} | {int((frames['valid'].PatientOrientation=='').sum())} | {int((frames['test'].PatientOrientation=='').sum())} |

字段分布见 `tables/metadata_value_counts.csv`。`Support Devices` 在三个 CSV 中均不存在。`PatientOrientation` 实际值为 Erect/Recumbent/空值；不能将其解释成正位或侧位。

## 键和标签一致性

训练/验证/测试列分别见 `tables/audit_manifest.json`。标签 CSV 行单位是 Image；统计时按 `(Subject_id, Study_id)` 去重。训练集同 Study 标签矛盾数 {sum(1 for x in label_conflicts if x['split']=='train')}，验证集 {sum(1 for x in label_conflicts if x['split']=='valid')}。测试集只有图像元数据，没有 `Predict_class`。 用户已确认训练/验证中未列出的类别为阴性；只有有效、非空标签列表适用该规则。

""", encoding="utf-8")

    label_table = label_stats_df.copy()
    for col in ["positive_share_among_explicit_binary_labels", "negative_share_among_explicit_binary_labels", "closed_world_candidate_positive_rate_if_absence_means_negative"]:
        label_table[col] = label_table[col].map(lambda x: fmt_num(x, 4))
    comparison_display = comparison[["class_id", "category", "train_listed_studies", "train_listed_code_share", "valid_listed_studies", "valid_listed_code_share", "valid_minus_train_listed_code_share"]].copy()
    for col in ["train_listed_code_share", "valid_listed_code_share", "valid_minus_train_listed_code_share"]:
        comparison_display[col] = comparison_display[col].map(lambda x: fmt_num(x, 4))
    comparison_md = markdown_table(comparison_display)
    (docs / "label_distribution.md").write_text(f"""# Study 级标签分布

## 解析规则与语义限制

训练/验证 CSV 使用 `Predict_class`，观察到的格式是 `1`、`10` 或 `7;9` 这类分号分隔整数 ID；ID 范围为 1–10。用户已确认：训练集和验证集中未列出的类别为阴性。源数据没有附相应标签字典，因此该语义依据项目提供的规则，而不是从图像推断。每个 Study 的多张图片标签必须一致；空值、格式错误或 Study 内冲突会阻止闭世界统计。测试集不含标签列，不用于标签比例或权重估计。

统计以 `(Subject_id, Study_id)` 去重。阳性数来自明确列出的 ID，阴性数为 Study 总数减去阳性数；无效/缺失标签不会被认定为阴性。当前训练/验证 CSV 中全部标签有效且同一 Study 标签一致，因此 `N_unknown=0`、`N_valid=Study总数`。

## Study 级正负样本数量

`tables/label_distribution_study_level.csv` 保存按已确认语义计算的 Study 级二元计数。每 Study 多图只算一次。

| 划分 | ID | 类别 | Study总数 | N_pos | N_neg | N_unknown | N_valid | 阳性比例 | 阴性比例 | N_neg/N_pos |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(f"| {r.split} | {r.class_id} | {r.category} | {int(r.study_count):,} | {int(r.N_positive_explicitly_listed):,} | {int(r.N_negative_under_confirmed_rule):,} | {int(r.N_unknown):,} | {int(r.N_valid):,} | {fmt_num(r.positive_share_among_explicit_binary_labels)} | {fmt_num(r.negative_share_among_explicit_binary_labels)} | {fmt_num(r.negative_positive_ratio)} |" for r in label_stats_df.itertuples())}

### 类别代码在 Study 中的比例（只描述明确列出的代码）

{comparison_md}

训练集类别频数图展示的是显式列出的阳性 Study 数：

![训练集每类 Study 代码数](figures/train_study_class_code_counts.png)

## 标签共现、阳性数及 No Finding

每个 Study 的类别代码数分布：`tables/study_positive_label_count_distribution.csv`。类别共现 Study 数及 `P(j 被列出 | i 被列出)` 见 `tables/label_cooccurrence.csv` 与 `tables/label_conditional_cooccurrence.csv`。以下热图为训练集 Study 去重后的代码共现数量：

![Study 级标签共现](figures/train_label_cooccurrence.png)

No Finding (ID 10) 与任一 ID 1–9 的显式共现、单独出现数在 `tables/no_finding_relationship.csv`。如果 No Finding 与其他病种代码同时出现，那是数据编码事实，但它是否属于矛盾还取决于官方定义；本报告不据此清洗标签。

## 训练/验证差异和可评测性

训练/验证每类 Study 级正负样本数和比例见上表；共现明细有 split 字段，可计算任意类别条件率差异。验证集十类目前均同时有阳性和阴性，因此 AUROC 在标签条件上可计算；实际 AUROC/AUPRC/Macro-F1 仍须取得模型预测后计算。

| 类别 | 极稀有标记（显式阳性编码 Study < 100） |
|---|---|
{chr(10).join(f"| {r.category} | {'是' if r.N_positive_explicitly_listed < 100 else '否'}（训练={int(r.N_positive_explicitly_listed):,}，验证={int(label_stats_df[(label_stats_df.split=='valid') & (label_stats_df.class_id==r.class_id)].N_positive_explicitly_listed.iloc[0]):,}） |" for r in label_stats_df[label_stats_df.split=='train'].itertuples())}

""", encoding="utf-8")

    bce_table = pd.DataFrame(conditional_weights)
    bce_md = markdown_table(bce_table[["class_id", "category", "N_pos_listed_studies", "N_not_listed_studies", "A_BCE_pos_weight_1", "B_raw_neg_pos_ratio_actual", "C_sqrt_ratio_actual", "D_cap5_actual", "D_cap10_actual", "D_cap20_actual"]], floatfmt=".4f")
    bce_hypothesis_md = markdown_table(bce_table[["class_id", "category", "B_raw_closed_world_hypothesis_only", "C_sqrt_closed_world_hypothesis_only", "D_cap5_closed_world_hypothesis_only", "D_cap10_closed_world_hypothesis_only", "D_cap20_closed_world_hypothesis_only"]], floatfmt=".4f")
    (docs / "bce_asl_analysis.md").write_text(f"""# Weighted BCE 与 ASL 方案分析

## 加权 BCE 候选权重

`BCEWithLogitsLoss(pos_weight=...)` 的 `pos_weight[c]=N_neg[c]/N_pos[c]` 只乘在该类正样本 BCE 项上，不乘负样本项。用户已确认训练集中未列出的类别为阴性，因此以下候选值按 train Study 计数计算；验证和测试不用于估权。

| ID | 类别 | N_pos(train) | N_neg(train) | A: BCE | B: raw | C: sqrt | D: cap5 | D: cap10 | D: cap20 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
{bce_md}

这些是待比较方案，不代表最优权重。若某类 N_pos=0，比例不可估计，应屏蔽该类损失并报告；极不均衡时优先比较平方根或较低 cap，避免小样本阳性梯度主导更新。

## 未知值 mask 和 Study 级归一化

如果数据出现空值、未知或不确定状态，不应把这些元素改成阴性。可使用 `reduction='none'` 计算逐元素 BCE，再乘有效标签 mask；先按每个 Study 的有效类别数归一化，再对有效 Study 求平均。当前 train/valid CSV 标签完整，因此闭世界配置使每个 Study 的十类均参与监督。多图 Study 只按 Study 聚合并计算一次损失；估权只使用 train Study，不使用测试标签。

## 稳定性和 Macro-AUROC

- 标准 BCE 是必要基线，优化稳定且作为概率校准参考。
- 原始负正比可能强力放大稀有阳性梯度，也可能提高方差、过拟合风险并改变概率校准。
- 平方根和 cap=5/10/20 限制梯度放大，适合作为对照；比例本身不能证明 Macro-AUROC 会提高。
- `pos_weight` 改变损失的正负梯度平衡，不保证排序改善。最终按验证 Macro-AUROC 和逐类 AUROC/AUPRC 比较。

## ASL 必要性与实验优先级

训练有 {int(split_summary_df.loc[split_summary_df.split=='train','study_count_composite_key'].iloc[0]):,} 个 Study；按确认规则，每类有较多阴性项。数据规模和阳性率支持把 ASL 列入正式消融，但标签频数不能证明负样本容易，也不能预言 AUC 提升。

ASL 通过 `gamma_neg` 下调容易负样本贡献；如果困难负样本提供有效边界信号，过强抑制可能损失这些信号，也可能放大误标影响。Macro-AUROC关注排序，降低容易负项权重不保证 AUC 上升。

**建议：ASL 列为建议做，不是必做。** 使用相同划分和训练预算，对照标准 BCE、平方根/截断 Weighted BCE，以及 ASL（初始 `gamma_pos=0, gamma_neg=4`）。不预设 ASL 优于 BCE；当前没有模型训练或 AUC 结果。

""", encoding="utf-8")

    (docs / "study_fusion_analysis.md").write_text(f"""# Study 级多图融合分析

## 数据依据

训练集多图 Study 占比 {(split_summary_df.loc[split_summary_df.split=='train','two_image_studies'].iloc[0] + split_summary_df.loc[split_summary_df.split=='train','three_plus_image_studies'].iloc[0]) / split_summary_df.loc[split_summary_df.split=='train','study_count_composite_key'].iloc[0]:.1%}；验证集 {(split_summary_df.loc[split_summary_df.split=='valid','two_image_studies'].iloc[0] + split_summary_df.loc[split_summary_df.split=='valid','three_plus_image_studies'].iloc[0]) / split_summary_df.loc[split_summary_df.split=='valid','study_count_composite_key'].iloc[0]:.1%}；测试集 {(split_summary_df.loc[split_summary_df.split=='test','two_image_studies'].iloc[0] + split_summary_df.loc[split_summary_df.split=='test','three_plus_image_studies'].iloc[0]) / split_summary_df.loc[split_summary_df.split=='test','study_count_composite_key'].iloc[0]:.1%}。完整图片数与视角组合见 `tables/study_image_counts_*.csv`、`tables/study_view_combinations.csv`。训练每个 Study 最多 {max(int(size_tables[s].image_count.max()) for s in SPLITS)} 张。多视角可能互补，但数据统计不能判定哪个融合最优。

## 五种候选

| 方法 | 计算 | 训练期是否 Study 级 | 优点 | 风险/成本 |
|---|---|---|---|---|
| A 概率平均 | 每图 sigmoid 后按类算术平均 | 可逐图训练后做后处理 | 最易实现；概率范围合法；多视角平滑单图噪声 | 对每张图同权；平均可能稀释只在某视角可见的阳性证据 |
| B 概率最大池化 | 每类取图片概率最大值 | 可逐图训练后做后处理 | 对任一视角强证据敏感 | 图越多误报机会越大；放大单图假阳性，且视图数量造成偏置 |
| C Logit 平均 | 先平均 logit，再 sigmoid | 可逐图训练后做后处理 | 平均 log odds；未过 sigmoid 的数值融合 | 与概率平均不等价；容易受置信度尺度影响，校准需验证 |
| D 特征平均/最大池化 | 共享 backbone 特征按 Study 池化后分类 | **需要** Study 级建模或重新训练头部 | 可学习前融合；允许特征互补 | 训练改造和显存/批处理复杂度增加；特征最大池化同样可能过度响应 |
| E Attention Pooling | 学习每张图权重，对特征加权和 | **需要** Study 级建模 | 可按数据学习视角重要性；对变长 bag 灵活 | 增加参数和训练复杂度；数据量/标签语义不足时易过拟合；推理需同组影像与聚合 |

概率平均是对概率求期望，logit 平均对应几何赔率融合，特征平均是在分类头前融合表示；三者非数学等价。A/B/C 能在逐图模型的预测缓存上对照，无需重新训练；D/E 通常要训练时将同一 Study 图像作为一个 bag，再基于 Study 标签优化。

## 建议路线

- **第一版基线：A 概率平均。** 每 Study 先取所有图像预测，按 10 类平均一次；验证时每 Study 仅保留一条标签/预测。它成本最低、语义易解释，可作为官方融合未规定时的明确提交策略。
- **第二版对照：C Logit 平均。** 与 A 使用同一图像级模型和预测缓存，检查 Macro-AUROC、逐类 AUROC/AUPRC及概率校准。B max 可作为敏感性分析，重点关注假阳性和图像数偏差。
- **后期 Attention：有条件再做。** 先看 A/C 的 Study 验证结果、视角互补性和多图 Study 占比。如果多图约占训练集 {(split_summary_df.loc[split_summary_df.split=='train','two_image_studies'].iloc[0] + split_summary_df.loc[split_summary_df.split=='train','three_plus_image_studies'].iloc[0]) / split_summary_df.loc[split_summary_df.split=='train','study_count_composite_key'].iloc[0]:.1%} 且多视角频繁，Attention 值得作为后期正式消融；否则新增训练复杂度可能不划算。该比例只是实施优先级依据，不是模型效果证据。

## Batch、泄漏和延迟测量

每个 batch 以 Subject+Study 为样本 bag。变长视图可采用 padding+`view_mask` 或扁平化所有 images 并附 `study_index`，编码后 `scatter_mean/max` 或 masked softmax attention 聚合；限制每 Study 图片上限时，抽样规则需固定并报告。标签先在 Study 去重，同一 Study 只产生一个目标向量和损失。

划分必须在 Subject 级完成；同一 Subject 的 Study 不跨 train/valid/test。当前观察到跨划分 Subject 重叠数为 {len(overlap_rows)}。审计覆盖数据已有划分，但不代表任意重新拆分自动无泄漏。

在 T4 评估时预热后同步 CUDA，分别记录：单张图前向耗时、整 Study 端到端耗时（含图像解码/预处理和 N 张 backbone 前向）、聚合耗时；报告 batch=1 的 p50/p95 和最大值，并固定输入分辨率/精度/线程数。官方 100 ms 计时单位尚未确定：需要确认按图像还是按 Study，以及是否含解码、预处理和多图前向，不能自行假定。多图 Study 的延迟通常随图片数增长。

""", encoding="utf-8")

    (docs / "analysis_summary.md").write_text(f"""# 分析总结与下一步实验建议

## 当前事实

1. 训练/验证/测试分别有 {int(split_summary_df.loc[split_summary_df.split=='train','study_count_composite_key'].iloc[0]):,}/{int(split_summary_df.loc[split_summary_df.split=='valid','study_count_composite_key'].iloc[0]):,}/{int(split_summary_df.loc[split_summary_df.split=='test','study_count_composite_key'].iloc[0]):,} 个 Study；标签统计单位为 `(Subject_id, Study_id)`。
2. 用户已确认训练集和验证集中未列出的 `Predict_class` 类别为阴性。空值、格式错误和 Study 内冲突不会按阴性处理；测试集没有标签列，也不参与估权。
3. 两条路线的 `configs/data.yaml` 使用 `label_semantics: closed_world`；当前配置以标准 BCE 作为首轮基线，Weighted BCE 和 ASL 均为后续消融。
4. 目前没有启动模型训练，因此没有 AUC 或性能提升结论。验证指标和融合方案须在 Study 级预测上实际比较。

## 下一步

1. 分别进入 `DenseNet121/` 和 `ConvNeXtTiny/`，安装对应依赖后运行 `python train.py`；两条路线共享只读 `cxr_first/`，各自产生 checkpoint 和日志。
2. 首轮采用标准 BCE、当前 Study 融合配置及相同 Subject 划分；随后逐项比较平方根/截断 Weighted BCE、ASL 和 Study 融合方法。
3. 每次实验记录 Macro-AUROC、逐类 AUROC/AUPRC、有效样本数、训练时间、显存峰值和推理时间；类别缺正或缺负时 AUROC 标为不可计算。
4. 100 ms 延迟究竟按图像还是 Study 计时仍需官方口径；实际性能、ASL效果和多图融合优劣需要模型实验验证。
""", encoding="utf-8")

    print(json.dumps({
        "output": str(out), "source_files": len(files), "image_audit": image_summary,
        "split_summary": split_summaries, "subject_overlap_pairs": len(overlap_rows),
        "missing_image_refs": int((~refs.exists).sum()), "unreferenced_images": len(unreferenced),
        "label_conflicts": len(label_conflicts), "study_id_non_global_collisions": len(duplicate_study_id),
    }, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
