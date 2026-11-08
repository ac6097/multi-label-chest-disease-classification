"""Sequential H2-B -> H2-C -> H4 training driver (single GPU process series).

Runs one experiment at a time to avoid the 8 GB GPU multi-process OOM seen when
three trainings were started together. Completed experiments are skipped by the
underlying trainer (training_status.json complete=true), so the driver can be
re-run after an interruption and will resume the current experiment from its
last checkpoint.
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path

from genopt_common import EXPERIMENTS, EXPERIMENT_ROOT, read_json
from genopt_train import train_experiment

SERIES = ("H2B", "H2C", "H4")


def status_path(experiment: str) -> Path:
    return EXPERIMENT_ROOT / EXPERIMENTS[experiment]["label"] / "training_status.json"


def main() -> None:
    results = {}
    for experiment in SERIES:
        path = status_path(experiment)
        if path.is_file() and read_json(path).get("complete"):
            print(f"[series] {experiment} already complete, skipping", flush=True)
            results[experiment] = {"skipped_complete": True}
            continue
        print(f"\n[series] ===== starting {experiment} =====", flush=True)
        try:
            results[experiment] = train_experiment(experiment)
            print(f"[series] {experiment} finished", flush=True)
        except Exception as error:  # keep the series alive for the next experiment
            record = {
                "experiment": experiment,
                "exception_type": type(error).__name__,
                "exception": str(error),
                "traceback": traceback.format_exc(),
            }
            results[experiment] = record
            print(f"[series] {experiment} FAILED: {type(error).__name__}: {error}",
                  flush=True)
        print(f"[series] {experiment} result: "
              f"{json.dumps(results[experiment], ensure_ascii=True, default=str)[:800]}",
              flush=True)
    complete = {}
    for experiment in SERIES:
        path = status_path(experiment)
        complete[experiment] = bool(
            path.is_file() and read_json(path).get("complete"))
    print(f"[series] training completion flags: {complete}", flush=True)
    if all(complete.values()):
        print("[series] all trainings complete; running H2/H4 analysis", flush=True)
        from genopt_h2_analysis import run_h2_analysis

        summary = run_h2_analysis(force=True)
        print("[series] analysis done: "
              f"{json.dumps(summary, ensure_ascii=True, default=str)[:1200]}",
              flush=True)
    else:
        print("[series] not all trainings complete; analysis skipped", flush=True)


if __name__ == "__main__":
    main()
