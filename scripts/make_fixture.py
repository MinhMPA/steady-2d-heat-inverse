#!/usr/bin/env python
"""Generate the synthetic 'measured' dataset deterministically.

The observation fixture must be reproducible: an unseeded draw silently changes
the inverse problem's target and makes stored results incomparable.

Usage:
    python scripts/make_fixture.py --seed 0 --sigma 1e-3 --nmesh 128
"""

# type imports
import argparse
import hashlib
import json
from pathlib import Path

# numerical imports
import numpy as np

# mpi imports
from mpi4py import MPI

# local imports
from forward_solver import SteadyHeat2DForwardSolver


def h_true(x):
    """Ground-truth thermal conductivity used to synthesize the observations."""
    return 1.0 + 6.0 * x[0] ** 2 + x[0] / (1.0 + 2.0 * x[1] ** 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sigma", type=float, default=1e-3)
    parser.add_argument("--nmesh", type=int, default=128)
    parser.add_argument("--q", type=float, default=1.0)
    parser.add_argument("--dbc", type=float, default=300.0)
    parser.add_argument(
        "--out", type=Path, default=Path("test_data/blackbox_output.xdmf")
    )
    args = parser.parse_args()

    fwd = SteadyHeat2DForwardSolver(
        nmesh=args.nmesh,
        mesh_type="quadrilateral",
        h=h_true,
        q=args.q,
        DBC_value=args.dbc,
    )
    fwd.solve()
    fwd.add_noise(mu=0.0, sigma=args.sigma, seed=args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fwd.export_xdmf(str(args.out))

    if MPI.COMM_WORLD.rank == 0:
        digest = hashlib.sha256(
            np.ascontiguousarray(fwd.T_obs.x.array, dtype=np.float64).tobytes()
        ).hexdigest()
        meta = {
            "seed": args.seed,
            "sigma": args.sigma,
            "nmesh": args.nmesh,
            "q": args.q,
            "DBC_value": args.dbc,
            "h_true": "1 + 6*x^2 + x/(1 + 2*y^2)",
            "T_obs_sha256": digest,
        }
        meta_path = args.out.with_suffix(".meta.json")
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
        print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
