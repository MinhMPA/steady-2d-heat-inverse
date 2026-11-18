# Conductivity inference: corrections after PR review

The review began at PR #2 revision `6d7338d3bed425de9a54fa59549f9af18fefaf7b`,
compared with `493e8dc09c4429a0a2c7cf265b3349dc5216ba61`. An independent reviewer
confirmed the nine findings recorded on 16 November and identified an additional
numerical-identity omission in private PETSc settings. The posterior and discrete
adjoint equations were consistent with the declared fixed-thermometer experiment.

## Corrections

1. Effective DOLFINx and FFCx settings now enter numerical identity. Each inverse
   problem owns a construction-time snapshot; later changes cannot relabel its
   existing prior or heat forms. Changed settings reject append; archives remain
   readable. Cache location, timeout and logging choices are excluded.
2. Private posterior heat/adjoint prefixes reject external PETSc overrides.
   Declared `HeatNumerics` settings govern both initial and continued chains.
   The process-wide options database is also recorded conservatively.
3. Temporary solver options are installed and restored with their complete
   names. In the qualified PETSc/DOLFINx combination, prefix-based deletion left
   actual entries behind. The correction preserves pre-existing options and
   restores them even when solver setup fails. Independent tangent equations
   use the same option cleanup while retaining their separate mathematics.
4. Pilot calculations stop admitting initialization spreads after memory or
   storage exhaustion. Calibration workers recognize SQLite resource codes,
   including chained causes, and preserve the stop across coordinator restart.
5. Closed chain workspaces release their gradient form and adjoint field even
   if the closed handle is retained; the parent keeps weak workspace references.
6. Legacy heat solves preserve a requested nonzero starting temperature. Private
   posterior solves retain their declared deterministic zero-start policy.
7. The notebook consumes the shared generation-aware report and resolves study
   paths from the detected project root. An unfinished extension cannot display
   the earlier result as its current posterior.
8. Benchmark totals count each attempted heat/adjoint solve, including line-search
   trials. Failed assembly cannot recount an earlier solve's iterations.
9. Real-transform spectra count conjugate Fourier modes with the appropriate
   multiplicity, including distinct odd/even Nyquist cases.

## Verification and interpretation

The first complete focused correction selection passed **61 tests**. Subsequent
checks additionally cover fresh-process user/local compiler configuration and
restoration of existing options after setup failure. The final complete-suite
and Linux CI results, with revision and links, are recorded in the
[PR opening message](https://github.com/MinhMPA/steady-2d-heat-inverse/pull/2).
The suite includes the long manufactured-Gaussian recovery checks, exact
same/fresh-process continuation, legacy adjoint checks and HMC notebook execution.
Long scientific study calculations marked `hmc_study` remain a separate task.

The independent reviewer compared odd/even random cross-spectra with complete
Fourier-plane averages: the largest absolute difference was `3.38e-14`.
Seven direct and seven chained resource-failure cases were classified correctly;
ordinary numerical failure remained distinct. A real SQLite page-limit failure
also exercises admission and restart through the calibration coordinator.

Recomputing the existing mesh-32 multifidelity notebook with corrected accounting
gave 3254 fixed-accuracy versus 1205 adaptive-accuracy linear iterations. Relative
conductivity errors were 0.1209 and 0.1275, respectively. Both optimizations ended
without convergence: the fixed run exhausted its iteration budget and the
adaptive run failed its line search. These observations do not establish a
general efficiency gain. In the small regression case, adaptive accuracy used
more total work; the old guaranteed-saving assertion was removed. The notebook
now reports the measured cost ratio and reconstruction accuracy explicitly.

The numerical corrections change source/continuation identity. Historical
calibration and mesh-study archives were preserved and are not migrated for
append. Passing implementation tests does not establish calibration, continuum
accuracy or adequacy of the Gaussian prior for sharp material interfaces. The
recorded 100-replica calibration and smooth/sharp mesh/prior studies remain
scientifically incomplete. No new production campaign was launched here.
