#!/usr/bin/env python
"""Validate a Study-level competition CSV without reading any labels from test."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = ["Subject_id", "Study_id", "Predict_class", "Probability"]
EXPECTED_CLASSES = set(range(1, 11))
KEY = ["Subject_id", "Study_id"]


def load_csv(path: Path) -> pd.DataFrame:
    # Keep all IDs as text, preserve leading zeros, and don't coerce blanks to NaN.
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)


def validate(test_csv: Path, prediction_csv: Path) -> list[str]:
    errors: list[str] = []
    test = load_csv(test_csv)
    pred = load_csv(prediction_csv)

    missing_test = [column for column in KEY if column not in test.columns]
    if missing_test:
        return [f"测试 CSV 缺少键列：{missing_test}"]
    if pred.columns.tolist() != REQUIRED_COLUMNS:
        errors.append(f"输出列必须按顺序恰为 {REQUIRED_COLUMNS}，实际为 {pred.columns.tolist()}。")
    missing_pred = [column for column in REQUIRED_COLUMNS if column not in pred.columns]
    if missing_pred:
        return errors + [f"预测 CSV 缺少列：{missing_pred}"]

    expected = test[KEY].drop_duplicates()
    if expected.duplicated(KEY).any():
        errors.append("内部错误：测试集联合键去重失败。")
    expected_keys = set(map(tuple, expected[KEY].itertuples(index=False, name=None)))

    if pred[KEY].eq("").any(axis=None):
        errors.append("Subject_id 或 Study_id 存在空值。")
    pred_keys = list(map(tuple, pred[KEY].itertuples(index=False, name=None)))
    actual_keys = set(pred_keys)
    missing_keys = sorted(expected_keys - actual_keys)
    extra_keys = sorted(actual_keys - expected_keys)
    if missing_keys:
        errors.append(f"遗漏测试 Study 联合键 {len(missing_keys)} 个，示例：{missing_keys[:5]}。")
    if extra_keys:
        errors.append(f"出现测试集外 Study 联合键 {len(extra_keys)} 个，示例：{extra_keys[:5]}。")

    class_numeric = pd.to_numeric(pred["Predict_class"], errors="coerce")
    if class_numeric.isna().any() or not class_numeric.map(lambda x: float(x).is_integer() if pd.notna(x) else False).all():
        errors.append("Predict_class 必须是整数类别 ID 1–10。")
    else:
        pred["__class"] = class_numeric.astype(int)
        invalid = sorted(set(pred["__class"]) - EXPECTED_CLASSES)
        if invalid:
            errors.append(f"Predict_class 含 1–10 以外的 ID：{invalid}。")

    probabilities = pd.to_numeric(pred["Probability"], errors="coerce")
    finite = probabilities.map(lambda x: math.isfinite(float(x)) if pd.notna(x) else False)
    if not finite.all():
        errors.append(f"Probability 有 {int((~finite).sum())} 个非有限或非数值项。")
    in_range = probabilities.between(0.0, 1.0, inclusive="both")
    if not in_range.all():
        errors.append(f"Probability 有 {int((~in_range).sum())} 项超出 [0,1] 或无法解析。")

    if "__class" in pred:
        sizes = pred.groupby(KEY, dropna=False).size()
        wrong_rows = sizes[sizes != 10]
        if len(wrong_rows):
            errors.append(f"有 {len(wrong_rows)} 个 Study 不是恰好 10 行。")
        duplicates = pred.duplicated(KEY + ["__class"], keep=False)
        if duplicates.any():
            errors.append(f"同一 Study 重复类别行数 {int(duplicates.sum())}。")
        bad_coverage = []
        for key, group in pred.groupby(KEY, dropna=False, sort=False):
            classes = set(group["__class"])
            if classes != EXPECTED_CLASSES:
                bad_coverage.append((key, sorted(classes)))
        if bad_coverage:
            errors.append(f"有 {len(bad_coverage)} 个 Study 未完整覆盖类别 1–10，示例：{bad_coverage[:5]}。")

    # Since the official output has exactly four columns, this also detects an exported index.
    if any(str(column).startswith("Unnamed:") for column in pred.columns):
        errors.append("发现额外索引列；CSV 不应输出 DataFrame index。")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-csv", type=Path, required=True, help="test.csv；仅读取 ID，不读取任何标签")
    parser.add_argument("--predictions", type=Path, required=True, help="待验证的 Study 级预测 CSV")
    args = parser.parse_args()
    errors = validate(args.test_csv, args.predictions)
    if errors:
        print("校验失败：")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print("校验通过：测试 Study 联合键完整；每个 Study 恰有 10 个互异类别 ID；概率有限且位于 [0,1]；列名与列顺序正确。")


if __name__ == "__main__":
    main()
