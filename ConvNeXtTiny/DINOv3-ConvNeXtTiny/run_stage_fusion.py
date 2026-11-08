"""CLI for the R8/R9/R10 controlled stage-fusion experiments.

Examples (run from ConvNeXtTiny/DINOv3-ConvNeXtTiny):

    python run_stage_fusion.py prepare --experiment R8
    python run_stage_fusion.py train   --experiment R8
    python run_stage_fusion.py train   --experiment R8 --analyze
    python run_stage_fusion.py analyze
    python run_stage_fusion.py status
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def _prepare(experiment: str) -> dict:
    import torch

    from stage_fusion_experiment import (
        EXPECTED_HEAD_SHA,
        EXPECTED_WEIGHT_SHA256,
        IMAGE_SIZE,
        OUTPUT_ROOT,
        SEED,
        build_run_configs,
        build_masked_datasets,
        capture_fixed_validation_batch,
        load_r5_contract,
        read_json,
        run_gradient_check,
        test_initialization_equivalence,
        write_json,
    )
    from dino_analysis import LABEL_NAMES

    contract = load_r5_contract()
    feature, config, freeze = build_run_configs(experiment, contract)
    from stage_fusion_experiment import GROUP_NAMES
    from stage_fusion_experiment import OUTPUT_ROOT as ROOT
    run_dir = ROOT / GROUP_NAMES[experiment]
    run_dir.mkdir(parents=True, exist_ok=True)
    source = (HERE / config["data"]["source_config"]).resolve()
    train_data, valid_data, data_config = build_masked_datasets(source, SEED, IMAGE_SIZE)
    if (len(train_data), int(sum(train_data.image_counts)), len(valid_data),
            int(sum(valid_data.image_counts))) != (18513, 30344, 859, 1422):
        raise RuntimeError("Train/Validation counts differ from the R5 manifest")
    if set(train_data.study_keys) & set(valid_data.study_keys):
        raise RuntimeError("Train/Validation Study keys overlap")
    runtime = contract["runtime"]

    equivalence_path = run_dir / "initial_equivalence_test.json"
    if equivalence_path.is_file():
        equivalence = read_json(equivalence_path)
    else:
        print(f"[prepare] {experiment}: CPU FP32 / CUDA BF16 initialization equivalence",
              flush=True)
        equivalence = test_initialization_equivalence(experiment, valid_data, runtime)
        write_json(equivalence_path, equivalence)
        write_json(run_dir / "initialization_equivalence_test.json", equivalence)
    gradient_path = run_dir / "gradient_check.json"
    if gradient_path.is_file():
        gradient_check = read_json(gradient_path)
    else:
        print(f"[prepare] {experiment}: real-batch delayed-gradient / memory preflight",
              flush=True)
        gradient_check = run_gradient_check(
            experiment, contract, config, freeze, runtime, train_data)
        write_json(gradient_path, gradient_check)
    summary = {
        "experiment": experiment,
        "run_dir": str(run_dir),
        "equivalence_passed": equivalence.get("passed"),
        "gradient_check_passed": gradient_check.get("passed"),
        "peak_reserved_gib": gradient_check["peak_reserved_gib"],
        "checks": gradient_check["checks"],
        "test_data_accessed": False,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def _train(experiment: str, analyze_after: bool) -> dict:
    from stage_fusion_experiment import run_experiment

    _prepare(experiment)
    status = run_experiment(experiment)
    result = {"training": status, "test_data_accessed": False}
    if analyze_after:
        from stage_fusion_analysis import analyze_group, _valid_data
        print(f"[analyze] {experiment}: unified inference, abalations, bootstrap, export",
              flush=True)
        valid_data = _valid_data()
        info = analyze_group(experiment, valid_data)
        result["analysis"] = {
            "run_dir": info["run_dir"],
            "unified_macro_auroc": info["bootstrap"]["models"][
                {"R8": "R8_Full", "R9": "R9_Full", "R10": "R10_Full"}[experiment]
            ]["macro_auroc"],
            "export": info["export"],
        }
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result


def _status() -> dict:
    from stage_fusion_experiment import GROUP_NAMES, OUTPUT_ROOT, read_json

    payload = {}
    for experiment, group in GROUP_NAMES.items():
        run_dir = OUTPUT_ROOT / group
        status_path = run_dir / "training_status.json"
        if not status_path.is_file():
            payload[experiment] = {"status": "not_started", "run_dir": str(run_dir)}
            continue
        status = read_json(status_path)
        payload[experiment] = {
            "status": "complete" if status.get("complete") else "in_progress",
            "completed_epochs": status.get("completed_epochs"),
            "best_epoch": status.get("best_epoch"),
            "best_macro_auroc": status.get("best_macro_auroc"),
            "stop_reason": status.get("stop_reason"),
            "run_dir": str(run_dir),
        }
    for group in ("R5_Global_Only_512", "R7_ResidualTransfer_512"):
        run_dir = OUTPUT_ROOT / group
        status_path = run_dir / "training_status.json"
        status = read_json(status_path) if status_path.is_file() else {}
        payload[group] = {
            "status": "complete" if status.get("complete") else "unknown",
            "best_epoch": status.get("best_epoch"),
            "best_macro_auroc": status.get("best_macro_auroc"),
        }
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",
                        choices=["prepare", "train", "analyze", "compare", "status"])
    parser.add_argument("--experiment", choices=["R8", "R9", "R10"])
    parser.add_argument("--groups", nargs="+",
                        choices=["R5", "R7", "R8", "R9", "R10"],
                        help="analyze only these groups, then rebuild comparisons")
    parser.add_argument("--analyze", action="store_true",
                        help="run per-experiment analysis right after training")
    arguments = parser.parse_args()
    if arguments.command == "status":
        _status()
        return
    if arguments.command == "prepare":
        if not arguments.experiment:
            parser.error("prepare requires --experiment")
        _prepare(arguments.experiment)
        return
    if arguments.command == "train":
        if not arguments.experiment:
            parser.error("train requires --experiment")
        _train(arguments.experiment, arguments.analyze)
        return
    if arguments.command == "compare":
        from stage_fusion_analysis import run_comparative_analysis
        result = run_comparative_analysis()
        print(json.dumps({
            "status": "comparative_complete",
            "experiments": list(result["metadata"]),
            "test_data_accessed": False,
        }, ensure_ascii=False), flush=True)
        return
    from stage_fusion_analysis import run_all_analysis
    run_all_analysis(experiments=arguments.groups)


if __name__ == "__main__":
    main()
