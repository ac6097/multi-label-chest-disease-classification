"""CLI for the G0/G1 (and, after explicit approval only, G2) diagnostics.

Examples (run from ConvNeXtTiny/DINOv3-ConvNeXtTiny):

    python run_generalization_diagnostics.py audit
    python run_generalization_diagnostics.py g0
    python run_generalization_diagnostics.py g1
    python run_generalization_diagnostics.py all     # G0 then G1, then stop

G2 is deliberately NOT runnable from this CLI: it requires explicit user
confirmation after reviewing g2_go_no_go_decision.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))


def _audit() -> dict:
    import collect_generalization_audit as audit

    audit.main()
    from generalization_common import DIAG_ROOT

    path = DIAG_ROOT / "environment_and_checkpoint_audit.json"
    if not path.is_file():
        raise RuntimeError("environment audit JSON was not produced")
    return {"audit_json": str(path)}


def _g0(force: bool) -> dict:
    from generalization_g0 import run_g0

    return run_g0(force=force)


def _g1(force: bool) -> dict:
    from generalization_common import G0_DIR, read_json

    g0_path = G0_DIR / "g0_summary.json"
    if not g0_path.is_file():
        raise RuntimeError("G1 requires a completed G0; run `g0` first")
    g0_summary = read_json(g0_path)
    if not g0_summary.get("gate_passed"):
        raise RuntimeError(
            "G0 gate did not pass; fix the inference issue before G1. "
            "See G0_Inference_Consistency/g0_final_report.md")
    from generalization_g1 import run_g1

    return run_g1(force=force)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["audit", "g0", "g1", "report", "all"])
    parser.add_argument("--force", action="store_true",
                        help="recompute cached inference passes")
    arguments = parser.parse_args()
    if arguments.command == "audit":
        result = _audit()
    elif arguments.command == "g0":
        result = _g0(arguments.force)
    elif arguments.command == "g1":
        result = _g1(arguments.force)
    elif arguments.command == "report":
        from generalization_g1 import run_reports_only

        result = run_reports_only()
    else:
        _audit()
        g0_summary = _g0(arguments.force)
        if not g0_summary.get("gate_passed"):
            print(json.dumps({
                "status": "g0_gate_failed",
                "message": "G1 not started; see G0 report",
                "test_data_accessed": False,
            }, ensure_ascii=False), flush=True)
            return
        result = {"g0_gate_passed": True, "g1": _g1(arguments.force)}
    print(json.dumps(result, ensure_ascii=True, indent=2, default=str))


if __name__ == "__main__":
    main()
