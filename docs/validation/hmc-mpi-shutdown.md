# MPI shutdown qualification, 11 November 2026

The dedicated macOS ARM64 posterior environment uses `FI_PROVIDER=tcp` before
Python starts. This is a measured configuration for MPICH 5.0.1 (`ch4:ofi`),
libfabric 2.6.0 and mpi4py 4.1.2. It preserves normal MPI finalization and leaves
errors visible. Follow the
[installation instructions](../installation.md#macos-mpich-transport-and-shutdown).

The change concerns the lifetime of the numerical process. It leaves the heat
equation, sensor likelihood, Gaussian log-conductivity prior, Hamiltonian flow
and Metropolis decision unchanged. Posterior calibration and chain convergence
remain separate requirements.

## What was observed

Earlier pilot processes wrote their numerical results and then reported an OFI
I/O error during `MPI_Finalize`, in the `flush_send_queue` path. The saved draws
alone therefore did not establish successful process completion.

Fresh, bounded experiments on 11 November imported only MPI, initialized one
rank with `THREAD_MULTIPLE`, passed a barrier and printed markers before and
after explicit finalization. Each process had an eight-second limit. No heat
solver or HMC library was imported.

| Interface selection | Sockets transport | TCP transport |
| --- | --- | --- |
| Normal provider selection | Finalized and exited 0 | Finalized and exited 0 |
| Existing `utun0` | Timed out after the pre-finalization marker | Finalized and exited 0 |
| Existing `utun3` | Timed out after the pre-finalization marker | Finalized and exited 0 |
| Existing `utun4` | Timed out after the pre-finalization marker | Finalized and exited 0 |

The paired logs confirm the requested provider and interface family. These
measurements isolate a transport-dependent shutdown stall without reproducing
the identical historical I/O error. They support the TCP configuration; they
do not establish that every earlier failure had the same cause. No network
interface was changed during the experiments. Timeout cleanup affected only
the experiment's own process.

The related [MPICH report](https://github.com/pmodels/mpich/issues/7760) describes
macOS finalization failures and successful use of `FI_PROVIDER=tcp`. In the
[MPICH 5.0.1 source](https://github.com/pmodels/mpich/blob/v5.0.1/src/mpid/ch4/netmod/ofi/ofi_init.c),
the sockets provider takes the `flush_send_queue` shutdown path before the
alternative collective-finalization branch. The
[libfabric sockets documentation](https://ofiwg.github.io/libfabric/v2.6.0/man/fi_sockets.7.html)
also identifies TCP as its replacement. These observations motivate selecting
an existing provider, without suppressing finalization or modifying MPICH.

## Equality of the physical calculation

Two independent processes used the unchanged numerical calculation from
`4853edcfebe6d22d5f2f64acb2d87c1e8989f696`, as present at documentation commit
`985458b91a2655e2ec6b4565b296ef7c0fa3a985`. One used sockets and one used TCP,
with normal interface selection. Both completed with exit code 0.

At `nmesh=16, 32, 64`, the test used nine fixed temperature sensors and the same
deterministic inferred field. All 30 saved comparisons were exactly equal:
inferred coordinates, conductivity, posterior potential, likelihood potential,
full posterior force, likelihood force, temperature field, sensor predictions,
and heat/adjoint iteration counts. Numerical-source hashes were unchanged
throughout the experiment. These are fixed-field comparisons, not convergence
or posterior-coverage measurements. Their wall times include different cache
states and are not a transport performance comparison.

## Permanent safeguards and review

The optional `tests/test_hmc_mpi_shutdown.py` check now follows ordinary mpi4py
initialization, requires `THREAD_MULTIPLE`, and verifies a barrier, explicit
finalization and successful exit in three fresh processes. Each process has a
30-second limit and inherits the caller's transport declarations. All three
passed with TCP in the dedicated environment.

New saved numerical identities include `FI_PROVIDER`, `FI_TCP_IFACE` and
`FI_SOCKETS_IFACE`, including unset values. Tests exercise setting, changing and
unsetting each declaration: an unchanged identity permits continuation, a
changed identity prevents continuation, and read-only access remains possible.
Reading archived draws does not require loading MPI. Source changes also alter
the numerical identity; earlier archives must retain their original record and
cannot be resumed under this changed source.

Independent scientific and implementation reviews found one reporting-memory
defect: a scalar chain summary retained every full conductivity and inferred
field while calculating statistics. The corrected summary retains only the
trajectory statistics and releases full fields before reading the next draw.
Regression checks cover complete, partial and empty histories across phases
and chains, preserving rejection repeats, counts, acceptance, divergences and
qualification decisions. The corrected code passed both scoped reviews.

## Verification record and limits

Both complete test processes exited with code 0 using TCP:

| Environment | Passed tests | Passed subtests | Skipped | Test duration |
| --- | --- | --- | --- | --- |
| Dedicated HMC, native qualification enabled | 630 | 32 | 0 | 2514.12 s |
| Original reconstruction environment | 456 | 32 | 29 | 1063.27 s |

These checks include the heat/adjoint calculation, Gaussian sampling experiments,
saved-chain restart and extension, independent numerical workers, archive access
and the notebook workflow. They use the normal test collection; the separately
scheduled scientific calibration and production studies are not included.
The HMC suite reported three warnings about XML-report metadata. PETSc also
printed 21 unused-option notices in the HMC environment and 18 in the original
environment. Neither process reported a finalization failure.

The source and tests were held fixed throughout both suites and checked against
saved SHA-256 hashes. Afterward, the only test change was the independently
reviewed `THREAD_MULTIPLE` refinement described above, whose exact bytes passed
the three separate fresh-process checks. The delivered source preserves the
full-suite hashes. Documentation was built with warnings treated as errors.

The local evidence directory is
`.superpowers/sdd/2026-10-26-full-field-blackjax-hmc/nov11-shutdown/`. It contains
the paired transport logs, physical arrays and comparisons, full test logs and
XML reports, independent review rulings, and the tested/delivered file hashes.

This qualification is specific to the measured machine and native-library
versions. Other installations need their own checks. Interface forcing is not
part of the recommended setup; local `lo0` experiments failed during MPI
initialization. Long production calculations must still record successful
process exit as well as saved trajectories.

The shutdown correction does not complete the outstanding 100-replica
simulation-based calibration, production chains or prior-sensitivity study.
The [scientific validation record](hmc-mesh-study.md) remains the source for
those release requirements.
