"""CLI for the H1/H2/H3 generalization-optimization experiments.

Commands (run from ConvNeXtTiny/DINOv3-ConvNeXtTiny):

    python run_genopt.py preflight
    python run_genopt.py calibrate
    python run_genopt.py train --experiment H1
    python run_genopt.py train --experiment H2
    python run_genopt.py train --experiment H3
    python run_genopt.py h0
    python run_genopt.py analyze
    python run_genopt.py all        # preflight + calibrate + H0 assembly +
                                    # H1/H2/H3 training + analysis
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parent
for _path in (str(PROJECT_DIR), str(HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from genopt_common import EXPERIMENT_ROOT, read_json  # noqa: E402


def _preflight():
    from genopt_preflight import run_preflight

    return run_preflight()


def _calibrate():
    from genopt_train import calibrate_l2sp

    return calibrate_l2sp()


def _train(experiment: str, force: bool):
    from genopt_train import train_experiment

    return train_experiment(experiment, force=force)


def _h3b_audit():
    from genopt_h3b_audit import run_audit

    return run_audit()


def _h2b_audit():
    from genopt_h2b_preflight import run_preflight

    return run_preflight()


def _r11_preflight():
    from r11_preflight import run_preflight

    return run_preflight()


def _r11_train(force: bool):
    from r11_train import train_r11

    return train_r11(force=force)


def _r11_analyze(force: bool):
    from r11_analysis import run_r11_analysis

    return run_r11_analysis(force=force)


def _h2_analyze(force: bool):
    from genopt_h2_analysis import run_h2_analysis

    return run_h2_analysis(force=force)


def _h3b_analyze(force: bool):
    from genopt_h3b_analysis import run_h3b_analysis

    return run_h3b_analysis(force=force)


def _h0():
    from genopt_analysis import assemble_h0

    return assemble_h0()


def _analyze(force: bool):
    from genopt_analysis import run_analysis

    return run_analysis(force=force)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=["preflight", "calibrate", "train", "h0", "analyze", "all",
                 "h3b_audit", "h3b_analyze", "h2b_audit", "h2_analyze",
                 "r11_preflight", "r11_train", "r11_analyze"])
    parser.add_argument("--experiment", choices=["H1", "H2", "H3", "H3B",
                                                 "H2B", "H2C", "H4"])
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    if arguments.command == "preflight":
        result = _preflight()
    elif arguments.command == "calibrate":
        result = _calibrate()
    elif arguments.command == "train":
        if arguments.experiment is None:
            raise SystemExit("--experiment is required for train")
        result = _train(arguments.experiment, arguments.force)
    elif arguments.command == "h0":
        result = _h0()
    elif arguments.command == "analyze":
        result = _analyze(arguments.force)
    elif arguments.command == "h3b_audit":
        result = _h3b_audit()
    elif arguments.command == "h3b_analyze":
        result = _h3b_analyze(arguments.force)
    elif arguments.command == "h2b_audit":
        result = _h2b_audit()
    elif arguments.command == "h2_analyze":
        result = _h2_analyze(arguments.force)
    elif arguments.command == "r11_preflight":
        result = _r11_preflight()
    elif arguments.command == "r11_train":
        result = _r11_train(arguments.force)
    elif arguments.command == "r11_analyze":
        result = _r11_analyze(arguments.force)
    else:
        preflight = _preflight()
        calibration = _calibrate()
        h0 = _h0()
        trainings = {}
        for experiment in ("H1", "H2", "H3"):
            trainings[experiment] = _train(experiment, arguments.force)
        analysis = _analyze(arguments.force)
        result = {"preflight": preflight, "calibration": calibration,
                  "h0": h0, "trainings": trainings, "analysis": analysis}
    print(json.dumps(result, ensure_ascii=True, indent=2, default=str))


if __name__ == "__main__":
    main()
