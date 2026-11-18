#!/usr/bin/env python
"""Run conductivity numerical qualification, calibration and physical mesh studies."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hmc_validation import (
    StudyConfig,
    run_numerical_pilot,
    run_sbc,
    make_truth_data,
    run_mesh_study,
    write_study_report,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("pilot", "sbc", "truth", "mesh"):
        command = commands.add_parser(name)
        command.add_argument("--config", required=True, type=Path)
        command.add_argument("--output", required=True, type=Path)
        if name == "pilot":
            command.add_argument("--data", type=Path)
        elif name == "truth":
            command.add_argument("--case", choices=("smooth", "sharp"), required=True)
        elif name == "mesh":
            command.add_argument("--data", type=Path, required=True)
            command.add_argument(
                "--nmesh", type=int, choices=(16, 32, 64), required=True
            )
            command.add_argument(
                "--total-draws",
                type=int,
                help="Explicit larger retained total per chain",
            )
            command.add_argument(
                "--reanalyze",
                action="store_true",
                help="Recompute evidence without changing saved draws",
            )
        else:
            command.add_argument("--replicates", type=int, default=100)
            command.add_argument("--workers", type=int, default=1)
    report = commands.add_parser("report")
    report.add_argument("--manifest", type=Path, nargs="+", required=True)
    report.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.command == "report":
        write_study_report(arguments.manifest, arguments.output)
        return 0
    config = StudyConfig.from_json(arguments.config)
    if arguments.command == "truth":
        print(make_truth_data(config, truth=arguments.case, output=arguments.output))
        return 0
    if arguments.command == "mesh":
        result = run_mesh_study(
            config,
            data_path=arguments.data,
            nmesh=arguments.nmesh,
            output=arguments.output,
            total_draws=arguments.total_draws,
            reanalyze=arguments.reanalyze,
        )
        return (
            0
            if result["status"] == "sampled" and result["cost_uncertainty_resolved"]
            else 2
        )
    if arguments.command == "pilot":
        result = run_numerical_pilot(config, arguments.output, data_path=arguments.data)
        return 0 if result["status"] == "numerical_pilot_complete" else 2
    result = run_sbc(
        config,
        arguments.output,
        replicates=arguments.replicates,
        workers=arguments.workers,
    )
    return 0 if result["calibration_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
