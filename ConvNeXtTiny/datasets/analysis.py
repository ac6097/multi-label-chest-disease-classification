"""CSV and image audit for the first step after dataset release."""
import argparse
from pathlib import Path

import pandas as pd
from PIL import Image


def _study_labels(frame, label_columns, study_column):
    values = frame[[study_column, *label_columns]].copy()
    for label in label_columns:
        numeric = pd.to_numeric(values[label], errors="coerce")
        values[label] = numeric.where(numeric.isin([0, 1]))
    return values.groupby(study_column, sort=False)[label_columns].first()


def summarize(csv_file, label_columns, output_dir, image_root="", image_column="image_path",
              study_column="study_id", subject_column="subject_id", view_column="view", other_csv=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(csv_file)
    required = [image_column, study_column, *label_columns]
    missing_columns = [column for column in required if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"CSV is missing required columns: {missing_columns}")

    study_labels = _study_labels(frame, label_columns, study_column)
    label_report = pd.DataFrame({
        "label": label_columns,
        "positive_studies": [(study_labels[label] == 1).sum() for label in label_columns],
        "negative_studies": [(study_labels[label] == 0).sum() for label in label_columns],
        "unknown_studies": [study_labels[label].isna().sum() for label in label_columns],
    })
    label_report.to_csv(output_dir / "label_distribution.csv", index=False)
    study_labels.fillna(0).T.dot(study_labels.fillna(0)).to_csv(output_dir / "label_cooccurrence.csv")

    study_sizes = frame.groupby(study_column, sort=False).size().rename("image_count")
    study_sizes.value_counts().sort_index().rename_axis("images_per_study").to_csv(
        output_dir / "study_image_counts.csv", header=["study_count"])
    if view_column in frame:
        frame[view_column].fillna("unknown").value_counts(dropna=False).rename_axis("view").to_csv(
            output_dir / "view_distribution.csv", header=["image_count"])

    image_root = Path(image_root) if image_root else Path(".")
    missing_images, image_formats, image_sizes, image_modes, gray_ranges = [], {}, {}, {}, {}
    for value in frame[image_column].dropna().astype(str):
        path = Path(value)
        if not path.is_absolute():
            path = image_root / path
        if not path.is_file():
            missing_images.append(str(path))
            continue
        try:
            with Image.open(path) as image:
                image_formats[image.format or "unknown"] = image_formats.get(image.format or "unknown", 0) + 1
                size_key = f"{image.width}x{image.height}"
                image_sizes[size_key] = image_sizes.get(size_key, 0) + 1
                image_modes[image.mode] = image_modes.get(image.mode, 0) + 1
                low, high = image.convert("L").getextrema()
                range_key = f"{low}-{high}"
                gray_ranges[range_key] = gray_ranges.get(range_key, 0) + 1
        except Exception:
            missing_images.append(str(path))

    no_finding = next((label for label in label_columns if label.strip().lower() == "no finding"), None)
    contradiction_studies = []
    if no_finding:
        disease_columns = [label for label in label_columns if label != no_finding]
        contradiction_studies = study_labels.index[
            (study_labels[no_finding] == 1) & (study_labels[disease_columns] == 1).any(axis=1)
        ].astype(str).tolist()

    subject_count = int(frame[subject_column].nunique()) if subject_column in frame else None
    subject_overlap = []
    if other_csv and subject_column in frame:
        other = pd.read_csv(other_csv)
        if subject_column in other:
            left = set(frame[subject_column].dropna().astype(str))
            right = set(other[subject_column].dropna().astype(str))
            subject_overlap = sorted(left & right)
            pd.DataFrame({"subject_id": subject_overlap}).to_csv(
                output_dir / "subject_overlap.csv", index=False)
    summary = {
        "csv": str(csv_file), "image_rows": len(frame), "studies": int(frame[study_column].nunique()),
        "subjects": subject_count, "other_csv": other_csv, "subject_overlap_count": len(subject_overlap),
        "missing_or_unreadable_images": missing_images, "image_formats": image_formats,
        "image_sizes": image_sizes, "image_modes": image_modes, "grayscale_ranges": gray_ranges,
        "no_finding_contradiction_studies": contradiction_studies,
    }
    pd.Series(summary, dtype=object).to_json(output_dir / "data_audit.json", force_ascii=False, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--labels", required=True, help="Comma-separated label columns in official order")
    parser.add_argument("--output-dir", default="outputs/reports/data_audit")
    parser.add_argument("--image-column", default="image_path")
    parser.add_argument("--study-column", default="study_id")
    parser.add_argument("--subject-column", default="subject_id")
    parser.add_argument("--view-column", default="view")
    parser.add_argument("--other-csv", default=None, help="Optional split CSV for Subject overlap audit")
    args = parser.parse_args()
    result = summarize(args.csv, [name.strip() for name in args.labels.split(",")], args.output_dir,
        args.image_root, args.image_column, args.study_column, args.subject_column, args.view_column,
        args.other_csv)
    print(result)


if __name__ == "__main__":
    main()
