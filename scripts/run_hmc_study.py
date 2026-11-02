#!/usr/bin/env python
"""Run the declared conductivity numerical pilot or discrete calibration study."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hmc_validation import StudyConfig, run_numerical_pilot, run_sbc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("pilot", "sbc"):
        command = commands.add_parser(name)
        command.add_argument("--config", required=True, type=Path)
        command.add_argument("--output", required=True, type=Path)
        if name == "pilot":
            command.add_argument("--data", type=Path)
        else:
            command.add_argument("--replicates", type=int, default=100)
    arguments = parser.parse_args()
    config = StudyConfig.from_json(arguments.config)
    if arguments.command == "pilot":
        result = run_numerical_pilot(config, arguments.output, data_path=arguments.data)
        return 0 if result["status"] == "numerical_pilot_complete" else 2
    result = run_sbc(config, arguments.output, replicates=arguments.replicates)
    return 0 if result["calibration_complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
