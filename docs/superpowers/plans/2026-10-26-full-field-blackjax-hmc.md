# Full-Field BlackJAX HMC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Infer the full conductivity posterior with BlackJAX split HMC, using the existing finite-element heat physics, a shared optimization objective, reproducible saved chains, and the agreed scientific validation.

**Architecture:** A frozen physical experiment supplies independent heat/adjoint workspaces and one posterior in Gaussian coordinates. A small, eagerly evaluated JAX connection supplies its adjoint derivative to BlackJAX; the project owns the exact Gaussian rotation, physical outputs, and atomic continuation records. Existing reconstruction calls keep their current objective and behavior.

**Tech Stack:** Python 3.12, DOLFINx 0.11, PETSc 3.25, NumPy/SciPy, CPU float64 JAX/BlackJAX, optional ArviZ Stats, and standard-library SQLite for transactional chain storage.

**Spec:** [Full-field scientific specification](../specs/2026-10-16-full-field-hmc-design.md), with [Q1–Q16](../specs/2026-10-17-hmc-requirements-interview.md), [Q17–Q24](../specs/2026-10-18-hmc-implementation-interview.md), and `docs/adr/0001`–`0003`. The request to write this plan follows the consolidated review; it does not authorize executing the plan in this turn.

## Global Constraints

- “Retain every conductivity degree of freedom on the chosen finite-element mesh; do not truncate the field to a few spatial modes.”
- “There is no mass matrix in this likelihood.” Measurements have fixed physical coordinates and known independent Gaussian errors, scalar or per sensor.
- `m = mu + tau**-1 A**-1 M_L**(1/2) z`, `A = kappa**2 M + K`, `h = h_ref exp(m)`, and `U = Phi + 0.5*z.T@z`; natural Neumann prior boundaries, positive `kappa`, `tau`, `h_ref`, no extra conductivity Jacobian or inherited bounds.
- “The kicks use the likelihood force” only; identity momentum covariance, exact Gaussian rotations, randomized durations, step-size-only warmup, fixed settings for retained draws.
- “Deterministic potential and force at a given `z`, independent of previous evaluations.” Fixed qualified PDE tolerances; never use the legacy adaptive tolerance controller here.
- “An unconverged finite-element solve stops the run with a diagnostic rather than being silently interpreted as zero posterior probability.” Nonfinite proposals repeat the preceding valid state and count as divergences.
- Each chain's heat/adjoint solve uses one MPI rank. Four chains by default; explicit single-chain checks remain possible. Counts are per chain and one master seed determines independent streams.
- Freeze physical inputs at problem construction, preserve the caller's pre-call `h,T`, and check mesh/source/boundaries before explicit application. Current `h,T` are excluded from the experiment comparison.
- Save every retained iteration, including rejection repeats, plus complete warmup/random/continuation state. Resume finishes the original requested total; extension explicitly raises that total. Changed physics or numerics cannot append to the chain.
- Return physical conductivity summaries and saved draws on demand, with chain/draw/node axes and spatial ordering; loading all draws is explicit.
- Main studies: `nmesh=16, 32, 64` (289, 1089, 4225 unknowns), fixed 256 interior sensors, `sigma=1e-3 K`, one finer-mesh data realization per truth, and the same physical prior across meshes.
- Qualification: four-chain rank-normalized split/folded `R-hat < 1.01`, combined bulk/tail ESS at least 400 for every monitored nonconstant quantity, and zero post-warmup divergences. Missing/nonfinite diagnostics cannot pass. Report Monte Carlo errors.
- Optimization returns the mode in log conductivity; physical means/quantiles are computed after transforming draws. Numerical correctness, sampling convergence, mesh approximation, and sharp-truth prior adequacy are separate conclusions.
- Preserve flat top-level imports and register modules in `pyproject.toml`; keep plotting and sampling dependencies lazy. No NumPyro, general NUTS, inferred hyperparameters, correlated noise, automatic cross-mesh field transfer, or distributed heat solves in this release.

## Review Focus

1. Uniform scalar conductivity followed by a nonuniform inferred field must retain all unknowns, preserve the caller during inference, and apply safely even if the caller originally used `fem.Constant` — Tasks 1 and 10.
2. Permuted/coincident sensors, fixed-temperature boundary sensors, nonuniform errors, and out-of-domain points must preserve their statistical meaning or fail explicitly — Task 2.
3. A prior with an inverse missing or with non-Gaussian coordinates must not silently lose auxiliary fields or acquire Gaussian energy — Tasks 3 and 5.
4. A process killed during a save, including uneven chain lengths or warmup, must recover a complete preceding transaction and resume exactly — Tasks 8 and 9.
5. A failed PDE solve midway through a trajectory must stop; a nonfinite force must instead reject and repeat the preceding valid state with the correct random stream and counters — Tasks 4, 6, 7, and 9.

---

## Starting evidence and execution conventions

Reviewed checkout: `feature/hmc-sampling`, HEAD `149f3a0`; the approved specifications are newer uncommitted documentation. Preserve the unrelated existing `CLAUDE.md`, `.claude/`, and `.DS_Store` changes. At execution, make the chosen worktree through `superpowers:using-git-worktrees` if isolation is wanted, and explicitly bring the reviewed plan/specification into it. Never silently execute against the older committed requirements. Stage exact files, never `git add .`.

Read-only metadata inspection on 2026-10-26 found `/Users/nguyenmn/miniconda3/envs/s2dhi-011/bin/python` at Python 3.12.13, DOLFINx 0.11.0, PETSc 3.25.4, NumPy 2.5.2, SciPy 1.18.0, h5py 3.16.0, pytest 9.1.1; JAX, BlackJAX, NumPyro and ArviZ were absent. The historical `29 passed, 4 subtests passed` baseline is not a fresh verification. No numerical tests or dependency installations were performed while writing this plan.

Use a dedicated `s2dhi-hmc` conda environment cloned from `s2dhi-011`, preserving that baseline environment. All `python` commands below mean its activated interpreter. Create the clone with `/Users/nguyenmn/miniconda3/bin/conda create -n s2dhi-hmc --clone s2dhi-011`, activate it, then run the current baseline suite and record source/environment identity. If MPI initialization is blocked by the execution sandbox, request the required tool escalation; do not weaken tests to avoid it. CPU qualification uses `JAX_ENABLE_X64=true`, `JAX_PLATFORMS=cpu`, `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`; ordinary execution may use other recorded thread counts, but exact-resume checks use the same environment and counts.

This is one connected inference feature, not independent subprojects: the likelihood, prior, sampler, and saved calculation must agree on the same field ordering and probability law. Tasks are independently reviewable; their interfaces below are binding. These are new planning choices within the accepted requirements; the quoted scientific constraints remain authoritative. Add each new source module to `py-modules` in its owning task. Test imports follow the existing `tests` package. All tests use real finite-element solves where physics is being checked; manufactured targets are used only where an exact sampling answer is required.

## File map and dependency order

| File | Responsibility |
| --- | --- |
| `src/heat_model.py` | Frozen discrete heat experiment, spatial ordering, independent forward workspaces, field conversion, experiment identity |
| `src/temperature_observations.py` | Fixed sensor data and P1/Q1 interpolation/transpose |
| `src/conductivity_prior.py` | Gaussian log-field specification, prior map/inverse/transpose and capability checks |
| `src/conductivity_posterior.py` | Sensor adjoint, likelihood/posterior value and forces, prediction, numerical failure evidence |
| `src/heat_inference.py` | Scientist-facing problem, complete estimates, optimization, sampling and continuation |
| `src/hmc_blackjax.py` | Eager supplied-derivative connection, split integrator, BlackJAX transition and step-size adaptation |
| `src/hmc_run.py` | Chain initialization, independent streams, progress and interruption orchestration |
| `src/posterior_store.py` | Immutable problem record and transactional chain/checkpoint storage |
| `src/posterior_diagnostics.py` | ArviZ Stats connection, physical observables and qualification |
| `src/posterior_samples.py` | Bounded-memory summaries, selected draws/temperatures, explicit application |
| `src/hmc_validation.py`, `scripts/run_hmc_study.py` | Reproducible numerical qualification, calibration, truth generation, pilots and mesh studies |
| Existing `forward_solver.py`, `domain_coefficient.py` | Narrow support for private discrete construction and explicit field application |
| Existing `adjoint_solver.py`, `tao_solver.py` | Only stale-form detection after explicit coefficient-handle replacement; legacy mathematics unchanged |
| `tests/test_hmc_*.py`, `tests/_hmc_helpers.py` | Tests owned by the corresponding task; helpers are test-private |
| `docs/hmc.md`, `docs/api/heat_inference.md`, `notebooks/HMCSampling.ipynb` | Equations, usable scientific workflow and interpretation |

Dependency order: 1 → 2/3 → 4 → 5/6 → 7 → 8 → 9 → 10 → 11 → 12 → 13 → 14. Task 12 assembles the public calculation and automated checks; Tasks 13–14 supply numerical calibration and scientific-study evidence.

## Numerical and persistence choices

Use **eager CPU BlackJAX** initially: a [JAX custom reverse derivative](https://docs.jax.dev/en/latest/_autosummary/jax.custom_vjp.html) exposes the host posterior value and its already-computed adjoint gradient; wrap initialization and each transition in [`jax.disable_jit()`](https://docs.jax.dev/en/latest/_autosummary/jax.disable_jit.html). Ordinary Python exceptions then remain usable around the heat solve. The [callback documentation](https://docs.jax.dev/en/latest/_autosummary/jax.pure_callback.html) treats callback exceptions as undefined behavior: do not use them or a hidden failure flag, and do not put PETSc into `jit`, `vmap`, or `pmap`. The first JAX task must prove this connection on the selected versions before it is used by HMC. This retains BlackJAX's transitions and adaptation; it is not a second native sampler. Measure its overhead in Task 13. A future compiled connection requires separate qualification and is outside this plan.

Candidate qualification pins: `blackjax==1.7.1`, `jax[cpu]==0.11.2`, `jaxlib==0.11.2`, `arviz-stats==1.3.3`. Add sampling and diagnostics as separate optional extras; the new route is qualified on Python 3.12+, while legacy Python support is not narrowed. These versions are implementation candidates, not a compatibility result. The [BlackJAX release](https://raw.githubusercontent.com/blackjax-devs/blackjax/1.7.1/pyproject.toml) requires JAX/JAXlib >=0.9; [JAXlib 0.11.2](https://pypi.org/project/jaxlib/0.11.2/) publishes a CPython 3.12 macOS ARM64 wheel. Preserve the conda numerical stack with constraints during resolution; reject a resolver proposal that silently changes DOLFINx/PETSc/NumPy/SciPy. Record all resolved versions, architecture, thread settings, and relevant source hashes in each run.

SQLite is the authoritative chain record: one transaction writes a retained/warmup record, the matching checkpoint, and counters. Use WAL with `synchronous=FULL`, one writer, no Python object pickles, and an exclusive run-directory creation. Store arrays as lossless `.npy` bytes with `allow_pickle=False`; metadata/state scalars are versioned JSON. This supplies rollback of incomplete writes instead of inventing transactions around an append-only HDF5 file. A run directory contains `problem.json`, `problem.npz`, `chains.sqlite`, and a small human-readable `status.json`; the last is advisory and recoverable from the database. Document that copying an active database alone is not a backup; use SQLite's backup operation or close the run first.

## Public calling convention to implement

These names become the planned public contract; the implementation tasks define their types.

```python
from heat_inference import HeatInverseProblem
from temperature_observations import TemperatureObservations
from conductivity_prior import GaussianLogConductivity

observations = TemperatureObservations(points=sensor_xy, values=d, sigma=sigma)
prior = GaussianLogConductivity(mu=mu, kappa=kappa, tau=tau, h_ref=h_ref)
problem = HeatInverseProblem(forward=fwd, observations=observations, prior=prior)
estimate = problem.optimize(initial_h=h0)
samples = problem.sample(initial=estimate, seed=42, warmup=1000, draws=2000,
                         chains=4, output="runs/conductivity-a")
summary = samples.summary()
h_draw = samples.field(chain=0, draw=0)
samples.apply_to(fwd, chain=0, draw=0)
samples = problem.resume("runs/conductivity-a")
samples = problem.resume("runs/conductivity-a", total_draws=3000)
```

Counts above illustrate the interface, not a promised converged length. `resume(output)` finishes the recorded target; `total_draws` explicitly extends it. `HeatInverseProblem.load(output)` reconstructs the frozen problem for use in a later session. It has no caller-owned forward model to mutate. File names, seed, solver settings and realized warmup are visible in `samples.metadata`; result interpretation never depends on BlackJAX object names.

### Task 1: Freeze the physical experiment and build independent heat workspaces

**Files:** Create `src/heat_model.py`, `tests/_hmc_helpers.py`, `tests/test_hmc_heat_model.py`; modify `src/forward_solver.py:34-135`, `src/domain_coefficient.py:28-130`, `pyproject.toml`.

**Interfaces:**
- Consumes: existing `SteadyHeat2DForwardSolver`, `fem.Function`, scalar P1/Q1 spaces and source/boundary data.
- Produces: `FrozenHeatModel.from_forward(forward) -> FrozenHeatModel`; fields `coordinates: ndarray[N,2]`, `cells: int64 ndarray`, `cell_type: str`, `source: ndarray[N]`, `boundary_dofs: int64 ndarray`, `boundary_values: ndarray`, `fingerprint: str`.
- Produces: `FrozenHeatModel.build_forward(*, prefix: str, numerics: HeatNumerics) -> SteadyHeat2DForwardSolver`; `coerce_field(value: FieldInput, model: FrozenHeatModel) -> ndarray[N]`; `canonical_dof_indices(V) -> ndarray[N]`, whose entries select native DOFs in canonical coordinate order.
- `FieldInput = float | Callable | fem.Function | NodalField`; `NodalField(values: ndarray, coordinates: ndarray)` carries ordering explicitly. Accept the existing UFL-style callable convention; bare arrays require pairing with coordinates. This conversion requires real finite values; positivity is checked specifically for conductivity, so a negative log-field mean remains valid.
- `HeatNumerics(rtol=1e-10, atol=1e-12, max_it=10000, ksp_type="cg", pc_type="hypre")` is frozen, finite and validated; these are development settings subject to Task 13 accuracy qualification.
- Test helper `make_forward(nmesh=4, cell="triangle", h=1.0, q=1.0, bottom=300.0)` returns the existing solver; `state_arrays(fwd)` returns copied `h,T` for conservation assertions.

- [ ] **Step 1: Establish the execution baseline.** In the dedicated environment run `python -m pytest -q --maxfail=1 --durations=5` and record its actual result before edits; inspect `git status --short` and record the source revision. A pre-existing failure is evidence to resolve, not a new test to delete.
- [ ] **Step 2: Write failing tests for physical identity and isolation.** Parameterize over triangle/quadrilateral. Add these assertions plus rejection of complex coefficients, nonfinite source/boundary data, and `mesh.comm.size != 1` before writing files or assembling new forms.

```python
def test_frozen_heat_model_survives_source_and_mesh_mutation():
    fwd = make_forward(q=2.0, bottom=315.0)
    frozen = FrozenHeatModel.from_forward(fwd)
    expected = frozen.fingerprint
    fwd.q.function.value = 9.0
    fwd.mesh.geometry.x[:, 0] *= 0.5
    clone = frozen.build_forward(prefix="hmc_test_a_", numerics=HeatNumerics())
    assert frozen.fingerprint == expected
    assert np.allclose(frozen.source, 2.0)
    assert np.allclose(frozen.boundary_values, 315.0)
    assert clone.h.function.x.array.size == 25
    assert np.isfinite(clone.solve().x.array).all()

def test_uniform_initialization_retains_all_field_unknowns():
    fwd = make_forward(h=2.0)
    before = state_arrays(fwd)
    frozen = FrozenHeatModel.from_forward(fwd)
    clone = frozen.build_forward(prefix="hmc_test_b_", numerics=HeatNumerics())
    values = coerce_field(2.0, frozen)
    assert values.shape == (25,)
    assert np.array_equal(values, np.full(25, 2.0))
    assert isinstance(clone.h.function, fem.Function)
    assert all(np.array_equal(a, b) for a, b in zip(before, state_arrays(fwd)))
```

- [ ] **Step 3: Run `python -m pytest tests/test_hmc_heat_model.py -q`.** Expected: missing new interface failure.
- [ ] **Step 4: Implement the snapshot and shared forward construction.** Canonicalize nodal ordering lexicographically by physical `(x,y)` and retain exact permutations to each FEM space. Copy connectivity, geometry, evaluated source, and actual boundary values; do not serialize callables or retain borrowed mutable arrays. Rebuild an independent affine P1/Q1 mesh on `MPI.COMM_SELF`; map by coordinates/connectivity, never by RBF interpolation. Check for ambiguous coordinates and unsupported elements explicitly.
- [ ] **Step 5: Add `SteadyHeat2DForwardSolver.from_discrete_data(*, domain, h_values, q_values, boundary_dofs, boundary_values, petsc_opts, prefix) -> SteadyHeat2DForwardSolver`.** Factor only its common heat-form/`LinearProblem` setup into `_initialize_linear_problem(prefix: str) -> None`, retaining an existing matching `T` object when rebuilding forms; the original constructor retains its coefficient conversion and defaults. Accept a matching scalar `fem.Function` in `BaseDomainCoefficient`, enabling the private constructor to create field-valued `h` before forms are built. Give every workspace a distinct PETSc prefix; start solves with zero initial guesses. Do not refactor the legacy adjoint objective.
- [ ] **Step 6: Run `python -m pytest tests/test_hmc_heat_model.py tests/test_smoke.py tests/test_domain_coefficient.py tests/test_solver_validation.py -q`.** Expected: all pass, including agreement between original and frozen heat predictions at `rtol=1e-10, atol=1e-10` in canonical ordering.
- [ ] **Step 7: Commit the exact Task 1 files:** `git commit -m "Add frozen heat experiments and private field workspaces"` after staging only the files listed above.

### Task 2: Represent fixed temperature observations and their transpose

**Files:** Create `src/temperature_observations.py`, `tests/test_hmc_observations.py`; modify `pyproject.toml`.

**Interfaces:**
- Consumes: a private forward's `V`, Task 1 coordinates/permutations.
- Produces: frozen `TemperatureObservations(points: ndarray[S,2], values: ndarray[S], sigma: float | ndarray[S])`; `from_function(T_obs: fem.Function, *, sigma) -> TemperatureObservations` copies original nodal coordinates and values.
- Produces: `SensorOperator(V, points)` with CSR `matrix`, `predict(T_dofs: ndarray[N]) -> ndarray[S]`, `transpose(weights: ndarray[S]) -> ndarray[N]` in native `V` ordering.

- [ ] **Step 1: Write the failing interpolation and input tests.** For both element types, evaluate native-DOF `T=300+2*x-3*y`; include interior, edge, vertex and duplicate points. Use DOLFINx point evaluation on the same field as an independent comparison. Parameterize invalid inputs: outside domain, zero/negative/NaN sigma, mismatched lengths, nonfinite points/data, and an empty sensor set; require `ValueError` with the offending sensor/index.

```python
def test_sensor_prediction_transpose_and_heteroscedastic_noise():
    fwd = make_forward()
    xy = np.array([[0.2, 0.3], [0.2, 0.3], [0.0, 0.0], [1.0, 1.0]])
    op = SensorOperator(fwd.V, xy)
    dofs = fwd.V.tabulate_dof_coordinates()
    T = 300.0 + 2*dofs[:, 0] - 3*dofs[:, 1]
    assert np.allclose(op.predict(T), 300+2*xy[:, 0]-3*xy[:, 1], atol=1e-12)
    w = np.array([1.0, -2.0, 3.0, 4.0])
    assert np.isclose(w @ op.predict(T), op.transpose(w) @ T, atol=1e-11)
    obs = TemperatureObservations(xy, op.predict(T), np.array([1., 2., 3., 4.]))
    xy[:] = -100.0
    assert np.array_equal(obs.points[0], [0.2, 0.3])
```

- [ ] **Step 2: Run `python -m pytest tests/test_hmc_observations.py -q`.** Expected: missing module/interface failure.
- [ ] **Step 3: Implement the declared interfaces.** Locate containing cells, pull physical points to reference cells, and tabulate the actual Basix P1/Q1 basis. Resolve shared-cell boundaries deterministically; reject extrapolation. Copy data in their supplied sensor order and preserve coincident independent measurements as separate rows. Boundary observations remain represented even when their predictions are analytically fixed. `from_function` does no interpolation of noisy values to a new mesh.
- [ ] **Step 4: Run the Task 2 test module.** Expected: all pass; add a permutation test asserting the permuted rows, values and sigmas give identical Gaussian misfit, and a nodal conversion test using a differently refined inference mesh with unchanged observation length/data.
- [ ] **Step 5: Commit the exact Task 2 files:** `git commit -m "Add fixed sensor observations and interpolation"`.

### Task 3: Implement the full Gaussian log-conductivity prior

**Files:** Create `src/conductivity_prior.py`, `tests/test_hmc_prior.py`; modify `pyproject.toml`.

**Interfaces:**
- Consumes: `FrozenHeatModel`, a private scalar space `V`, `coerce_field` for coordinate-associated inputs.
- Produces: `GaussianLogConductivity(*, mu: float | Callable | NodalField, kappa: float, tau: float, h_ref: float)` and `.bind(model, V) -> BoundGaussianPrior`.
- `BoundGaussianPrior`: `dimension: int`, `coordinate_schema={"z": (dimension,)}`, `coordinate_law="standard_normal"`, `to_log_h(z) -> ndarray[N]`, `to_h(z) -> ndarray[N]`, `from_h(h) -> ndarray[N]`, `pullback(z, grad_log_h) -> ndarray[N]`, `energy(z) -> float`, `grad_energy(z) -> ndarray[N]`, `record() -> dict`. Arrays use canonical ordering at the interface.
- Define a narrow `ConductivityPrior` protocol that declares its coordinate schema/law and forward/force/record capabilities; inverse initialization is an explicit optional capability. The first numerical implementation accepts only the Gaussian `{"z": (dimension,)}` schema. Preserve complete states as data, and reject unsupported sampling schemas/laws explicitly. A future implementation can have `dimension != N`; do not infer dimension from conductivity size.

- [ ] **Step 1: Write failing tests with an independently assembled dense covariance on `nmesh=2`.** Expose no production dense covariance. Test positive/finite parameter checks, Neumann constant mode, inverse and transpose, both element types, and scalar/callable/nodal mean fields.

```python
def test_full_prior_map_inverse_and_transpose():
    fwd = make_forward(nmesh=2)
    model = FrozenHeatModel.from_forward(fwd)
    prior = GaussianLogConductivity(mu=0.3, kappa=2., tau=1.5, h_ref=2.).bind(model, fwd.V)
    z = np.linspace(-0.4, 0.4, prior.dimension)
    assert prior.dimension == 9
    assert np.allclose(prior.from_h(prior.to_h(z)), z, atol=2e-12)
    J = np.column_stack([prior.to_log_h(e)-prior.to_log_h(np.zeros(9)) for e in np.eye(9)])
    v = np.arange(9, dtype=float)
    assert np.allclose(prior.pullback(z, v), J.T @ v, atol=2e-12)
    assert prior.energy(z) == pytest.approx(0.5*z@z)
    assert np.array_equal(prior.grad_energy(z), z)
```

- [ ] **Step 2: Run `python -m pytest tests/test_hmc_prior.py -q`.** Expected: missing prior interface failure.
- [ ] **Step 3: Implement sparse maps exactly as in the spec.** Assemble `M,K` without temperature boundary elimination; `M_L=diag(M@ones)`, `A=kappa**2*M+K`, factor `A` once with SciPy sparse LU, use transpose solves in `pullback`. The inverse is `tau*M_L**(-1/2)*A@(log(h/h_ref)-mu)`. Never form a production dense covariance, clip `h`, divide by dimension, or add a conductivity Jacobian. Fingerprint evaluated means and prior parameters separately from initial states.
- [ ] **Step 4: Run Task 3 tests.** Expected: all pass, including `J@J.T` versus the independently assembled `tau**-2*A^-1*M_L*A^-T` to `rtol=1e-11`, and positive variance of the constant mode. Reject nonpositive initial `h` explicitly; exponential overflow is classified as a nonfinite proposal later.
- [ ] **Step 5: Commit the exact Task 3 files:** `git commit -m "Add the full Gaussian log conductivity prior"`.

### Task 4: Compute the sensor posterior and discrete-adjoint force

**Files:** Create `src/conductivity_posterior.py`, `tests/test_hmc_posterior.py`; extend `tests/_hmc_helpers.py`; modify `pyproject.toml`.

**Interfaces:**
- Consumes: `FrozenHeatModel`, `TemperatureObservations`, `BoundGaussianPrior`, `SensorOperator`, `HeatNumerics`.
- Produces: `ConductivityPosterior(model, observations, prior, numerics, *, prefix)`; read-only `.prior`, `.observations`; `.evaluate(z: ndarray) -> PosteriorEvaluation`; `.predict_h(h: ndarray) -> tuple[ndarray, ndarray]` returns canonical full temperature and sensor predictions; `.close() -> None` and context management release owned workspaces.
- `PosteriorEvaluation(z, h, phi, grad_phi, potential, gradient, sensor_temperature, heat_iterations, adjoint_iterations)` contains copied float64 arrays and scalars; `potential=phi+0.5*z@z`, `gradient=grad_phi+z`. Read-only `.solve_counts` returns `SolveCounts(heat, adjoint, heat_iterations, adjoint_iterations)`, counting actual attempts independently of the probability law.
- `HeatSolveFailure(stage, position, reason, residual_norm, iterations)` and `NonfiniteProposal(stage, position)` are distinct exceptions. Invalid initial states are errors before starting warmup, not saved rejected draws.
- Helper `make_posterior(nmesh=4, cell="triangle", numerics=None) -> ConductivityPosterior` builds a deterministic test experiment: sensors at the 3-by-3 grid with axis coordinates `0.25,0.5,0.75`, noiseless data from nodal `h=1+0.2*x+0.1*y` on that mesh, `sigma=0.05`, `mu=0`, `kappa=2`, `tau=1`, `h_ref=1`, source 1, boundary 300.

- [ ] **Step 1: Write failing full-posterior and likelihood-only force tests.** Central differences use directions of unit Euclidean norm and steps `1e-2,1e-3,1e-4,1e-5`; compare at the best resolved step with relative error `<1e-5`. Require a Taylor-remainder slope between 1.8 and 2.2 over a preselected resolved interval. Test different `sigma_i`, boundary sensors, and both elements.

```python
def test_posterior_contains_sensor_likelihood_and_one_gaussian_energy():
    posterior = make_posterior()
    z = np.linspace(-0.1, 0.1, posterior.prior.dimension)
    ev = posterior.evaluate(z)
    r = (ev.sensor_temperature-posterior.observations.values)/posterior.observations.sigma
    assert ev.phi == pytest.approx(0.5*r@r)
    assert ev.potential == pytest.approx(ev.phi+0.5*z@z)
    assert np.allclose(ev.gradient, ev.grad_phi+z, atol=1e-12)
    posterior.evaluate(-z)
    again = posterior.evaluate(z)
    assert np.array_equal(again.gradient, ev.gradient)
    assert again.potential == ev.potential

def test_failed_heat_solve_is_not_a_nonfinite_proposal():
    posterior = make_posterior(numerics=HeatNumerics(rtol=1e-14, max_it=1, pc_type="none"))
    with pytest.raises(HeatSolveFailure) as exc:
        posterior.evaluate(np.zeros(posterior.prior.dimension))
    assert exc.value.stage == "heat"
    assert exc.value.iterations <= 1
```

- [ ] **Step 2: Run `python -m pytest tests/test_hmc_posterior.py -q`.** Expected: missing posterior interface failure.
- [ ] **Step 3: Implement the sensor adjoint by composition.** Use the private forward matrix with homogeneous adjoint Dirichlet values and RHS `B.T@((B*T-d)/sigma**2)`; zero constrained RHS entries, never add a mass matrix or the old roughness term. Assemble `-grad(T)·grad(lambda)*v*dx`, convert to canonical ordering, then pull back `h*grad_h` through the prior. Set zero KSP starting guesses, fixed tolerances and distinct prefixes; count solves outside the mathematical value. Store actual PETSc reasons/residuals on failures. Pre-check finite positions and `h`, including positive representability, before PDE assembly. Release PETSc objects deterministically on close/fatal error; a recoverable nonfinite proposal must leave the workspace usable for the next trajectory.
- [ ] **Step 4: Run Task 4 tests and `tests/test_grad_finitediff.py tests/test_grad_forwarddiff.py tests/test_grad_taylorexp.py`.** Expected: all pass. Add direct tangent-linear verification on the new sensor objective, using a private test helper in `_hmc_helpers.py`, and a same-position/different-history comparison. The legacy three gradient tests must keep their old objective.
- [ ] **Step 5: Commit the exact Task 4 files:** `git commit -m "Add the shared sensor posterior and adjoint force"`.

### Task 5: Optimize the same declared posterior and retain complete inferred states

**Files:** Create `src/heat_inference.py`, `tests/test_hmc_optimization.py`; extend `tests/_hmc_helpers.py`; modify `pyproject.toml`.

**Interfaces:**
- Consumes: Tasks 1–4; SciPy `optimize.minimize(method="L-BFGS-B", jac=True)` with no bounds.
- Produces: `HeatInverseProblem(*, forward, observations, prior, numerics: HeatNumerics | None=None)`; `.posterior: ConductivityPosterior`; read-only `.problem_id: str`; `.forward` is the supplied forward reference (or `None` after loading); `.optimize(*, initial_h: FieldInput, maxiter=500, gtol=1e-6, ftol=1e-12) -> PosteriorEstimate`.
- `InferredState(coordinates: dict[str, ndarray], problem_id: str)` uses `{"z": ...}` for this model; preserve every coordinate entry when carrying/saving it. `PosteriorEstimate(state, conductivity, objective, success, message, mode_coordinates="log_conductivity")` keeps failed optimization status visible.
- `HeatInverseProblem.initial_state(value: PosteriorEstimate | InferredState | FieldInput) -> InferredState` validates experiment/coordinate schema and only invokes a prior inverse for physical-only input.
- `HeatInverseProblem._new_posterior(*, prefix: str) -> ConductivityPosterior` constructs an independent workspace from the frozen inputs; `.close()` and context management release owned workspaces, never the caller's forward model.
- Helper `make_problem(nmesh=4, cell="triangle") -> HeatInverseProblem` uses the same test experiment as `make_posterior` and keeps its caller forward accessible to tests.

- [ ] **Step 1: Write failing tests of common probability law and state conservation.** Use the real small heat problem plus an analytic Gaussian objective for exact optimum checks. Use a test-only declared prior-schema fixture with an auxiliary coordinate and no inverse: full-state handoff preserves it; physical-only initialization raises `ValueError` explaining the missing inverse. The first sampler rejects this unsupported schema before starting; no production hierarchical prior is added. Also test a supported Gaussian-schema fixture with its inverse capability removed: an existing complete `z` state still initializes correctly.

```python
def test_optimization_handoff_preserves_posterior_and_caller():
    problem = make_problem()
    before = state_arrays(problem.forward)
    result = problem.optimize(initial_h=1.0)
    ev = problem.posterior.evaluate(result.state.coordinates["z"])
    assert result.objective == pytest.approx(ev.potential)
    assert result.mode_coordinates == "log_conductivity"
    assert np.allclose(result.conductivity, ev.h)
    assert problem.initial_state(result).problem_id == result.state.problem_id
    assert all(np.array_equal(a, b) for a, b in zip(before, state_arrays(problem.forward)))

def test_complete_state_needs_no_inverse_and_keeps_auxiliary_fields(future_problem):
    # This local test fixture declares z and an auxiliary interface coordinate;
    # its prior has no inverse and is deliberately unsupported by first-release HMC.
    state = InferredState({"z": np.zeros(9), "interface": np.array([0.6])},
                          future_problem.problem_id)
    carried = future_problem.initial_state(state)
    assert np.array_equal(carried.coordinates["interface"], [0.6])
    with pytest.raises(ValueError, match="inverse"):
        future_problem.initial_state(1.0)
```

- [ ] **Step 2: Run `python -m pytest tests/test_hmc_optimization.py -q`.** Expected: missing public problem/estimate interface failure.
- [ ] **Step 3: Implement the declared interfaces and immutable problem identity.** Combine canonical physical data, observation order/values/errors, prior record and coordinate schema into `problem_id`. Numerical settings have a separate identity used for continuation. Optimization minimizes `U(z)` and labels its constant-determinant log-field interpretation. Do not silently optimize before sampling or reuse `SteadyHeat2DTAOSolver._objgrad`. Use private workspaces; preserve caller state on success, optimizer failure and exceptions. A nonconverged estimate is returned with `success=False`, not relabelled as a posterior mode achieved.
- [ ] **Step 4: Run Task 5 tests and `tests/test_tao_postconditions.py`.** Expected: all pass, including frozen observations/source after caller mutation and exact retention of auxiliary coordinates. A nonfinite optimization trial ends optimization with `success=False`, the last finite state and an explicit message; do not invent a finite gradient or claim convergence. A failed PDE propagates as `HeatSolveFailure`.
- [ ] **Step 5: Commit the exact Task 5 files:** `git commit -m "Expose optimization of the declared conductivity posterior"`.

### Task 6: Qualify the supplied-adjoint connection to eager CPU JAX

**Files:** Create `src/hmc_blackjax.py`, `tests/test_hmc_jax_connection.py`, `constraints/hmc-cpu-py312.txt`; modify `pyproject.toml`.

**Interfaces:**
- Consumes: `value_and_grad(z: ndarray) -> tuple[float, ndarray]`, supplied by Task 4's evaluation, plus its two failure classes.
- Produces: `make_logdensity(value_and_grad) -> Callable` with value `-U` and supplied derivative `-grad_U`; `require_sampling_environment() -> dict` validates real float64, CPU and optional dependency versions before any run directory is created.
- The returned function supports eager `jax.value_and_grad` under `jax.disable_jit()`. Whole-chain `jit`, `vmap`, GPU and differentiating through PETSc are outside this release and must fail clearly when encountered.

- [ ] **Step 1: Add the qualification environment and failing connection tests.** In the dedicated clone, install the candidate pins listed above using constraints that preserve the inspected conda numerical packages. Add `sampling` and `diagnostics` extras separately. On Python below 3.12, expose a clear unsupported-sampling message while preserving legacy imports. Record the successful resolver output; do not treat a dependency import as evidence of a working connection.

```python
def test_connected_value_and_force_equal_direct_adjoint():
    posterior = make_posterior()
    z = np.linspace(-0.1, 0.1, posterior.prior.dimension)
    direct = posterior.evaluate(z)
    def value_and_grad(x):
        ev = posterior.evaluate(x)
        return ev.potential, ev.gradient
    logdensity = make_logdensity(value_and_grad)
    with jax.disable_jit():
        value, force = jax.value_and_grad(logdensity)(jnp.asarray(z))
    assert value.dtype == jnp.float64
    assert float(value) == direct.potential * -1
    assert np.array_equal(np.asarray(force), -direct.gradient)
```

- [ ] **Step 2: Run `python -m pytest tests/test_hmc_jax_connection.py -q`.** Expected: missing connection failure, rather than skipped tests in the qualified environment.
- [ ] **Step 3: Implement `make_logdensity` using `jax.custom_vjp`.** Evaluate the NumPy/PETSc value and gradient together once in its forward rule; retain the gradient for its reverse rule. Convert concrete CPU arrays at this boundary. Guard unsupported tracers before conversion. Keep PDE exceptions in the eager Python calling sequence; do not place raising code in `jax.pure_callback`, whose exception behavior is undefined. Make no global change to the caller's JAX configuration; require float64/CPU explicitly and use a scoped `disable_jit` for sampler operations.
- [ ] **Step 4: Run the connection tests with a counted analytic function and the real posterior.** Assert exactly one posterior evaluation per combined value/gradient request; deterministic repeated results; correct sign; and unchanged `HeatSolveFailure`/`NonfiniteProposal` types, including an exception on a later force evaluation. Add a negative test for attempted `jit` and for float32 configuration in isolated subprocesses. If eager BlackJAX qualification in Task 7 fails, repair this same connection or report the incompatibility; do not substitute another sampler or change the target.
- [ ] **Step 5: Commit the exact Task 6 files:** `git commit -m "Connect the finite element adjoint to eager JAX"`.

### Task 7: Implement and verify BlackJAX split Hamiltonian trajectories

**Files:** Extend `src/hmc_blackjax.py`; create `tests/test_hmc_dynamics.py`, `tests/test_hmc_gaussian.py`; extend `tests/_hmc_helpers.py`.

**Interfaces:**
- Consumes: Task 6's `logdensity`; BlackJAX `IntegratorState`, `hmc.init`, `hmc.build_kernel`, and `dual_averaging_adaptation`.
- Produces: `gaussian_split_integrator(logdensity_fn, kinetic_energy) -> Callable[[IntegratorState, float], IntegratorState]`, matching BlackJAX's integrator factory signature.
- `HMCControls(initial_step_size=0.1, target_acceptance=0.8, trajectory_interval=(pi/4, 3*pi/4), adapt_step_size=True, step_size_bounds=(1e-6, pi/8), divergence_threshold=1000.0, initial_spread=0.3)` is frozen. The spread and numerical defaults are provisional until Task 13; all realized settings are recorded.
- `BlackJAXChain(value_and_grad, controls)` exposes `initialize(z) -> HMCState`, `transition(state, *, key, step_size, num_steps) -> Transition`, `begin_warmup() -> AdaptationState`, `adapt(adaptation, acceptance_probability) -> AdaptationState`, `final_step_size(adaptation) -> float`.
- `Transition(state, acceptance_probability, accepted, divergent, delta_energy, initial_energy, final_energy, num_steps, failure_reason)` copies the new or repeated state; unavailable energy values and the ordinary no-failure reason use `None`. `HMCState` is BlackJAX's position/logdensity/gradient state; `AdaptationState` contains every field returned by the pinned BlackJAX adaptation routine. Only Task 7 knows these library representations. Module functions `encode_state(state) -> dict`, `decode_state(fields) -> HMCState`, `encode_adaptation(state) -> dict`, `decode_adaptation(fields) -> AdaptationState` provide lossless named-field conversion for storage.
- Test helper `gaussian_value_and_grad(mean, covariance)` returns the exact quadratic negative log density and force; its likelihood part is this quadratic minus `0.5*z@z`.

- [ ] **Step 1: Write failing Hamiltonian tests.** Use a correlated 3-dimensional Gaussian and several starting momenta. Test the prior-only target separately; test real-PDE stepping through the same connection as Task 6. No replacement trajectory routine is used by the public sampler.

```python
def test_prior_only_rotation_conserves_energy_and_reverses():
    logp = make_logdensity(lambda z: (0.5*z@z, z.copy()))
    step = gaussian_split_integrator(logp, lambda p: 0.5*jnp.dot(p, p))
    with jax.disable_jit():
        start = IntegratorState(jnp.array([0.2, -0.3]), jnp.array([1., 0.4]),
                                jnp.array(-0.065), jnp.array([-0.2, 0.3]))
        end = step(start, 0.2)
        back = step(end, -0.2)
    assert np.allclose(back.position, start.position, atol=2e-14)
    assert np.allclose(back.momentum, start.momentum, atol=2e-14)
    H0 = -start.logdensity + 0.5*jnp.dot(start.momentum, start.momentum)
    H1 = -end.logdensity + 0.5*jnp.dot(end.momentum, end.momentum)
    assert abs(float(H1-H0)) < 2e-14
```

- [ ] **Step 2: Run `python -m pytest tests/test_hmc_dynamics.py tests/test_hmc_gaussian.py -q`.** Expected: missing trajectory interfaces.
- [ ] **Step 3: Implement likelihood kicks and exact rotations.** Recover the likelihood force from the stored gradient as `grad_phi=-state.logdensity_grad-state.position`. Apply half kick, the exact `(z,p)` rotation, evaluate the new full log density/gradient, then the second half kick. Use [`blackjax.hmc.build_kernel`](https://blackjax-devs.github.io/blackjax/_modules/blackjax/mcmc/hmc.html)`(integrator=gaussian_split_integrator)` with an all-ones inverse mass vector; BlackJAX owns momentum refreshment, endpoint Metropolis acceptance and rejected-state selection. Run initialization and each transition inside `jax.disable_jit()`.
- [ ] **Step 4: Implement warmup and trajectory controls.** Use [BlackJAX dual averaging](https://blackjax-devs.github.io/blackjax/autoapi/blackjax/adaptation/step_size/index.html) with `t0=10, gamma=0.05, kappa=0.75`, target 0.8, no mass adaptation; fix the averaged step size after warmup. Cap proposed step sizes at the recorded upper bound; fail with evidence if the required size falls below the lower bound. Manual controls may disable adaptation and set the step size explicitly. Require `0 < Tmin < Tmax < pi`, `eps_max <= min(Tmin/2, pi-Tmax)` so rounding cannot produce a prior resonance. Task 9 draws `T` uniformly and uses `L=max(1, floor(T/eps+0.5))`; record `T,L,L*eps`.
- [ ] **Step 5: Implement failure/energy reporting.** Compute initial Hamiltonian from the incoming state and BlackJAX's sampled momentum; compare with the proposal Hamiltonian. Flag finite `abs(delta_energy)>1000` as divergent while retaining ordinary Metropolis behavior. Nonfinite value/force/Hamiltonian returns an explicit rejected repeated state, acceptance probability 0 and divergence; catch only `NonfiniteProposal`, not `HeatSolveFailure`. When a full energy cannot be formed, store a reason and null energy, not a fictitious finite number. The host driver preallocates the next random key so this path is reproducible.
- [ ] **Step 6: Run the Task 7 tests.** Require reversibility to `1e-10`; a small finite-difference phase-space Jacobian determinant within `1e-6` of one; energy-error ratios consistent with order two over a fixed duration at `eps=0.1,0.05,0.025`; prior-only acceptance `1` within `1e-12`; exact rejection repeats. For seeded independent and correlated Gaussian targets use four chains and compare every mean/covariance entry with the analytic value within five batch-estimated Monte Carlo standard errors (and check those errors against chain length). Also inject a mid-trajectory PDE failure and require it to propagate. These are scientific acceptance tests of the selected library connection.
- [ ] **Step 7: Commit the exact Task 7 files:** `git commit -m "Add prior preserving split HMC with BlackJAX"`.

### Task 8: Save one complete physical calculation transactionally

**Files:** Create `src/posterior_store.py`, `tests/test_hmc_store.py`; extend `src/heat_inference.py`; modify `pyproject.toml`.

**Interfaces:**
- Consumes: frozen model/observations/prior records, all inferred coordinates, and Task 7 state/adaptation field mappings. Serialization never depends on pickling library objects.
- Produces: `ProblemRecord(metadata: dict, arrays: dict[str, ndarray], problem_id: str, numerical_id: str)`; `HeatInverseProblem.to_record() -> ProblemRecord`; `HeatInverseProblem.from_record(record) -> HeatInverseProblem`; `HeatInverseProblem.load(output: PathLike) -> HeatInverseProblem`.
- `ChainCheckpoint(chain, phase, warmup_done, draws_done, inferred_coordinates, physical_state, hmc_state, rng_state, adaptation_state, step_size, counters)` holds the next-iteration random state and every continuation field, including retained `h`/sensor predictions in `physical_state` so resume needs no extra evaluation of the current state; `TrajectoryRecord(chain, phase, index, inferred_coordinates, h, sensor_temperature, statistics)` contains the preceding completed transition's outcome. Both use explicit named arrays/scalars, including auxiliary coordinates.
- `RunStore.create(path, *, problem: ProblemRecord, run_metadata: dict) -> RunStore`; `.open(path, *, writable=False)`; `.initialize_chains(checkpoints: Sequence[ChainCheckpoint])`; `.commit_trajectory(record, checkpoint)`; `.checkpoint(chain)`; `.records(chain, *, phase="sample", start=0, stop=None)` yields records; `.update_target(total_draws)`; `.write_failure(evidence)`; `.backup(destination)`; `.close()`. Read-only `.metadata` and `.problem_record` serve readers. Initial checkpoints are committed together, once, before any warmup.

- [ ] **Step 1: Write failing serialization and transaction tests.** Use a complete synthetic checkpoint fixture containing arrays, a future auxiliary coordinate, warmup adaptation and nonzero diagnostic counters. Reject object arrays, nonfinite model inputs, unsupported schema versions, attempts to overwrite a run, and corrupt problem hashes.

```python
def test_atomic_draw_and_checkpoint_survive_process_death(tmp_path):
    # The test's child process commits draw 0, begins draw 1's transaction,
    # writes its row but exits via os._exit before committing its checkpoint.
    output = tmp_path / "interrupted"
    run_store_crash_child(output)
    with RunStore.open(output) as store:
        rows = list(store.records(chain=0))
        assert [row.index for row in rows] == [0]
        assert store.checkpoint(0).draws_done == 1
        assert np.array_equal(rows[0].inferred_coordinates["z"],
                              store.checkpoint(0).inferred_coordinates["z"])
```

- [ ] **Step 2: Run `python -m pytest tests/test_hmc_store.py -q`.** Expected: missing storage interface. Define `run_store_crash_child(path)` locally in this test module with a subprocess; do not simulate durability only by raising a Python exception.
- [ ] **Step 3: Implement the SQLite record and frozen-problem reconstruction.** Create the directory exclusively, write immutable metadata/arrays using temporary files and atomic replacement, then mark initialization complete. Use schema-versioned SQLite tables with unique `(chain,phase,index)` and one transaction for trajectory, checkpoint and counters. Validate lengths/indices before commit. WAL/FULL synchronization and a single-writer lock prevent concurrent resume. Keep interrupted creation distinguishable from a valid empty run; never silently erase it. Persist source identity, dependency versions, floating-point mode, linear solver controls, field ordering, observation units, initialization, counts and controls.
- [ ] **Step 4: Implement continuation identity checks and backups.** Compare exact problem identity and numerical environment before opening for append. Numerical source identity hashes the relevant numerical modules; record the Git revision separately so unrelated documentation edits do not invalidate continuation. Count extensions may change only the requested retained total. Use `Connection.backup` for portable snapshots and reject an existing backup destination. `status.json` can lag a committed transaction and must be rebuilt from authoritative database counts. Normal context-manager close releases locks.
- [ ] **Step 5: Run Task 8 tests.** Expected: all pass, including abrupt death during warmup, rejected-state repeats, checkpoint/record index disagreement rejection, round-trip array identity, backup reopening, competing writer rejection, and reconstruction of matching heat predictions from a fresh process. No caller or original Python callable is needed to reconstruct the problem.
- [ ] **Step 6: Commit the exact Task 8 files:** `git commit -m "Save complete HMC trajectories and continuation states atomically"`.

### Task 9: Run independent chains and reproduce interrupted calculations

**Files:** Create `src/hmc_run.py`, `src/posterior_samples.py`, `tests/test_hmc_continuation.py`; extend `src/heat_inference.py`, `tests/_hmc_helpers.py`, `tests/test_hmc_optimization.py`; modify `pyproject.toml`.

**Interfaces:**
- Consumes: Tasks 4–8. A separate private posterior workspace belongs to each chain; sequential chain execution is the first supported scheduling choice.
- Produces: `HeatInverseProblem.sample(*, initial=None, seed: int, warmup=1000, draws=2000, chains=4, output: PathLike | None=None, controls: HMCControls | None=None, initial_states: Sequence | None=None, progress: Callable[[Progress], None] | None=None) -> PosteriorSamples`.
- Exactly one of `initial` and `initial_states` is required. Entries accept the same types as `initial_state`; explicit starts must match `chains`. `seed` is a validated unsigned 32-bit integer. Counts are positive integers, with zero warmup allowed only for explicitly disabled adaptation and an explicit step size.
- `HeatInverseProblem.resume(output, *, total_draws: int | None=None, progress=None) -> PosteriorSamples` completes the saved target or extends it. Supplying a smaller total is an error; no seed/initialization/tuning replacement is accepted.
- `run_chains(problem, *, initial_states, request: RunRequest, store: RunStore, progress=None) -> None` in `hmc_run.py`; `RunRequest(warmup, draws, chains, seed, controls)` records controls. `Progress(chain, phase, completed, target, elapsed_seconds, output)` is emitted after a committed boundary.
- Initial `PosteriorSamples.open(output)` exposes `output: Path`, `metadata: dict`, `.close()` and context management; Task 10 adds physical access. A default output is a unique `runs/hmc-<UTC>-<uuid>` directory. Return only after the requested work finishes; interruptions/failures leave a loadable partial run and propagate with its path in the saved status/evidence.

- [ ] **Step 1: Write failing continuation tests on a small real heat posterior.** A test-only `interrupt_after(chain, phase, completed)` creates a progress callback that raises `KeyboardInterrupt` after that committed boundary. Test both warmup and retained draws, then compare uninterrupted and resumed arrays, random/adaptation states, rejection histories and counters exactly.

```python
def test_resume_after_warmup_interruption_matches_uninterrupted(tmp_path):
    problem = make_problem(nmesh=2)
    common = dict(initial=1.0, seed=42, chains=1, warmup=8, draws=12)
    problem.sample(output=tmp_path/"whole", **common)
    before = state_arrays(problem.forward)
    with pytest.raises(KeyboardInterrupt):
        problem.sample(output=tmp_path/"split", progress=interrupt_after(0, "warmup", 3), **common)
    assert all(np.array_equal(a, b) for a, b in zip(before, state_arrays(problem.forward)))
    problem.resume(tmp_path/"split")
    assert_runs_equal(tmp_path/"whole", tmp_path/"split")
```

- [ ] **Step 2: Run `python -m pytest tests/test_hmc_continuation.py -q`.** Expected: missing orchestration/public sampling interface. Define `assert_runs_equal(a,b)` in `_hmc_helpers.py` to compare scientific records/checkpoints exactly, excluding wall times, paths, interruption messages and separately recorded work spent in aborted trajectories. Committed solve/diagnostic counters must match exactly; actual total computational cost also records aborted/replayed work.
- [ ] **Step 3: Implement initialization and random streams.** Use JAX `threefry2x32` keys; fold the master key by chain index and separately by purpose for initialization/trajectories. Save key data and implementation. Multiple chains centered on one estimate use independent `z_center+s*xi`; one chain uses its supplied state exactly. Explicit per-chain starts are used exactly. Check and record initial states before warmup; never retry until a favorable dispersion is found. Unsupported prior coordinates fail before launching the sampler; test `future_problem.sample(initial=complete_auxiliary_state, seed=1)` raises `ValueError` matching `unsupported.*coordinates` before creating an output directory. No initialization or dispersion is repeated on continuation.
- [ ] **Step 4: Implement one completed-trajectory loop.** Split each saved key into next-state, duration and HMC keys before evolving; draw the duration, choose the step count, take the transition, update warmup once, then commit its state/records/counters together. Cache the accepted evaluation by exact position to avoid storing proposal predictions after a rejection; if unavailable, evaluate the retained position before committing, with its extra solve cost counted. Save physical `h` and sensor predictions for every retained state, including repeats. Release a chain workspace after it finishes. Keep imports of orchestration/results inside the public methods where needed to avoid circular module dependencies.
- [ ] **Step 5: Implement stop, resume and extension behavior.** On a PDE failure, save chain/phase/trajectory, proposed position, PETSc reason, residual and iterations; preserve the preceding checkpoint and re-raise. On an uncommitted interrupt, resume replays from that checkpoint and its key. On a post-commit interrupt, the next iteration follows the committed key. Preserve the caller by using private workspaces throughout. A changed total is atomically recorded; completed warmup and prior draws stay immutable. Validate identity before append, including dependency/source and thread settings relevant to exact continuation.
- [ ] **Step 6: Run Task 9 tests.** Include the accepted uneven-count example `[2000,2000,1370,0]` using fast analytic transitions and an additional smaller real-PDE analogue; resume to 2000 and extend to 3000 without changing existing draws. Require four independent chains by default, reproducible dispersed starts, explicit single-chain behavior, different scheduling order yielding identical per-chain results, and rejection repeats on injected nonfinite proposals. Test PDE failure after several valid force calls and compare resumed random/diagnostic histories. Require a clear rejection of changed observations, source, prior, mesh, tolerances or environment before any append.
- [ ] **Step 7: Commit the exact Task 9 files:** `git commit -m "Run reproducible independent HMC chains with exact continuation"`.

### Task 10: Return physical fields and apply a selected field explicitly

**Files:** Extend `src/posterior_samples.py`, `src/forward_solver.py`, `src/adjoint_solver.py`, `src/tao_solver.py`; create `tests/test_hmc_results.py`, `tests/test_hmc_apply.py`.

**Interfaces:**
- Consumes: Task 8 records, Task 9 `PosteriorSamples`, Task 1 canonical coordinates/permutations and physical fingerprints.
- Produces: `PosteriorSamples.coordinates`, `.cells`, `.field(*, chain, draw) -> ndarray[N]`, `.state(*, chain, draw) -> InferredState`, `.load_draws(*, chains=None, start=0, stop=None) -> ndarray[C,D,N]` for explicitly requested equal-length selections.
- `.summary(*, quantiles=(0.05,0.5,0.95), node_chunk_size=64) -> FieldSummary(mean, sd, quantiles, probabilities, coordinates, counts, qualified)`; `.sensor_predictions(*, chain, start=0, stop=None) -> ndarray[D,S]`; `.temperature(*, chain, draw) -> ndarray[N]` recomputes a selected full field privately.
- `.as_function(*, chain=None, draw=None, summary=None) -> fem.Function` creates a matching private mesh/function for existing plotting functions. Exactly one complete draw selection or a physical `FieldSummary` statistic is required; `summary` is `"mean"`, `"sd"` or `("quantile", probability)`.
- `.apply_to(forward, *, chain=None, draw=None, summary=None) -> fem.Function` requires exactly one selected draw/summary, validates the physical experiment, applies positive conductivity and solves its matching temperature. Standard deviation alone is available for plotting but cannot be applied as a conductivity summary.
- `SteadyHeat2DForwardSolver.apply_conductivity_field(values: ndarray, *, coordinates: ndarray) -> fem.Function` assigns the nodal conductivity and solves its temperature transactionally; both application methods return that temperature. A read-only `coefficient_generation: int` changes only when a coefficient handle is replaced; existing adjoint/TAO objects record it and reject stale forms before solving or computing gradients.

- [ ] **Step 1: Write failing physical-summary tests with deliberately unequal log and physical averages.** Construct saved draws from known `z` values, including repeated states; calculate expected `h` summaries directly. Use bounded node chunks and make an accidental call to full-chain loading fail in this test.

```python
def test_saved_physical_summary_includes_rejection_repeats(saved_known_run):
    samples, expected_h = saved_known_run
    result = samples.summary(node_chunk_size=2)
    assert np.allclose(result.mean, expected_h.mean(axis=(0, 1)))
    assert np.allclose(result.quantiles,
                       np.quantile(expected_h, [0.05, 0.5, 0.95], axis=(0, 1)))
    assert np.array_equal(samples.field(chain=0, draw=1), expected_h[0, 1])
    assert np.array_equal(result.coordinates, samples.coordinates)
    assert not result.qualified  # No diagnostics have been computed yet.

def test_apply_promotes_constant_and_rejects_changed_source(saved_known_run):
    samples, _ = saved_known_run
    fwd = make_forward(nmesh=2, h=1.0)
    samples.apply_to(fwd, chain=0, draw=0)
    assert isinstance(fwd.h.function, fem.Function)
    assert np.allclose(coerce_field(fwd.h.function, FrozenHeatModel.from_forward(fwd)),
                       samples.field(chain=0, draw=0))
    fwd.q.function.value = 9.0
    before = state_arrays(fwd)
    with pytest.raises(ValueError, match="source"):
        samples.apply_to(fwd, chain=0, draw=1)
    assert all(np.array_equal(a, b) for a, b in zip(before, state_arrays(fwd)))
```

- [ ] **Step 2: Run `python -m pytest tests/test_hmc_results.py tests/test_hmc_apply.py -q`.** Expected: missing result/application methods. Define `saved_known_run` locally using Task 8's public storage interface, on the `make_problem(nmesh=2)` experiment, with positive nonconstant fields and at least one rejection repeat.
- [ ] **Step 3: Implement physical access in bounded memory.** Compute pooled physical mean and population standard deviation over every retained state; exact quantiles load all draws for only the requested node chunk. Preserve the chain axis for diagnostics. Uneven partial runs expose counts and clearly labelled pooled summaries, with `qualified=False`; whole-array requests for unequal lengths require an explicit common selection and never silently truncate. Warmup is excluded; no retained draws produces a clear error for summaries. Field/draw indices are zero-based and range-checked; validate positive chunk sizes and probabilities in `[0,1]`. Attach immutable coordinates/connectivity to fields; full temperature evaluation uses a reconstructed private posterior and never changes the caller. For existing plots use `f = samples.as_function(...)` then `plot_scalar_mesh(f.function_space, f.x.array, "h")`, so VTK coordinates follow the function DOFs; test this spatial association with a linear field using `dolfinx.plot.vtk_mesh(V)`. Explain that `T(mean(h))` and `mean(T(h))` are distinct.
- [ ] **Step 4: Implement explicit application as a reversible operation.** Compare the destination's mesh/connectivity, evaluated source and boundary data with the saved model before changing anything; compare by canonical physical order, not object identity. Exclude its current `h,T`. For existing field-valued `h`, preserve its coefficient handle. For a constant, prepare a field coefficient and new forward forms, solve successfully, then adopt them and increment `coefficient_generation`. Restore old handles, arrays, forms and generation on failure; preserve the `T` object that legacy objects may reference. Add generation checks to legacy adjoint `solve`/`assemble_gradient` and TAO `solve`/`_objgrad`; stale objects must tell the user to reconstruct them. No legacy objective or normal constructor semantics change.
- [ ] **Step 5: Run both Task 10 modules and the full existing suite.** Require success for a scalar-initialized destination; mismatched source/boundaries/mesh leaves `h,T` byte-identical; changing only `h,T` is allowed; selected temperature agrees with a fresh solve. Test a forced application solve failure, stale legacy adjoints after constant promotion, and continued normal legacy calls when the coefficient handle is unchanged. Check predictions and plotting conversions preserve caller state on failure as well as success.
- [ ] **Step 6: Commit the exact Task 10 files:** `git commit -m "Expose saved physical fields and explicit model application"`.

### Task 11: Qualify physical posterior summaries with established diagnostics

**Files:** Create `src/posterior_diagnostics.py`, `tests/test_hmc_diagnostics.py`; extend `src/posterior_samples.py`; modify `pyproject.toml`.

**Interfaces:**
- Consumes: stored physical conductivities/sensor predictions, frozen mesh mass weights, counts and divergences; optional `arviz_stats` array interfaces.
- Produces: `PosteriorSamples.diagnostics(*, node_chunk_size=64) -> DiagnosticReport`; `DiagnosticReport(status, qualified, per_quantity, worst_locations, divergences, counts, package_version, reasons)`.
- `status` is `qualified`, `unqualified`, `incomplete` or `unavailable`. Per-quantity records contain split/folded rank R-hat, bulk/tail ESS, mean MCSE and MCSE of the reported quantile endpoints. `FieldSummary.qualified` consults a matching complete diagnostics record, never merely the presence of four chains.

- [ ] **Step 1: Write failing threshold and physical-observable tests.** Test rank/folded statistics against direct calls to the pinned package, not a second homemade estimator. Isolate policy in `qualify_diagnostics(*, chains, complete, rhat, bulk_ess, tail_ess, divergences, analytically_fixed) -> tuple[bool, list[str]]` so exact boundaries have deterministic tests.

```python
def test_qualification_thresholds_are_strict_and_missing_values_fail():
    base = dict(chains=4, complete=True, rhat=np.array([1.009]),
                bulk_ess=np.array([400.]), tail_ess=np.array([400.]),
                divergences=0, analytically_fixed=np.array([False]))
    assert qualify_diagnostics(**base)[0]
    assert not qualify_diagnostics(**(base | {"rhat": np.array([1.01])}))[0]
    assert not qualify_diagnostics(**(base | {"tail_ess": np.array([399.9])}))[0]
    assert not qualify_diagnostics(**(base | {"rhat": np.array([np.nan])}))[0]
    assert not qualify_diagnostics(**(base | {"divergences": 1}))[0]
    assert not qualify_diagnostics(**(base | {"chains": 1}))[0]
```

- [ ] **Step 2: Run `python -m pytest tests/test_hmc_diagnostics.py -q`.** Expected: missing diagnostic interface.
- [ ] **Step 3: Implement the ArviZ Stats connection.** Pass arrays with chain axis 0 and draw axis 1; compute the maximum of rank-normalized split and folded-split [R-hat](https://python.arviz.org/projects/stats/en/latest/api/generated/arviz_stats.rhat.html), combined bulk/tail [ESS](https://python.arviz.org/projects/stats/en/latest/api/generated/arviz_stats.ess.html), and mean/quantile [MCSE](https://python.arviz.org/projects/stats/en/latest/api/generated/arviz_stats.mcse.html) through the package. Never sum individual-chain ESS. Monitor every nodal `h`, FEM-area-weighted mean conductivity, left-minus-right and bottom-minus-top area-weighted contrasts, and every sensor prediction. Store worst values and physical node/sensor locations. Evaluate in bounded variable chunks without discarding chain ordering or draws.
- [ ] **Step 4: Implement explicit qualification status.** Require at least four complete chains, all finite nonconstant diagnostics, strict `R-hat<1.01`, both ESS values `>=400`, and zero post-warmup divergences. Identify prescribed-temperature sensors analytically from the boundary operator and exempt only those fixed quantities; accidental constant chains cannot pass. A missing diagnostics extra yields `unavailable` and an installation hint. Uneven/incomplete runs can report diagnostics on an explicitly labelled common prefix for inspection, but cannot qualify the full requested calculation. Save reports with the exact draw counts and invalidate them after extension.
- [ ] **Step 5: Run Task 11 tests.** Include separated chains, highly correlated draws, constant inferred nodes, known fixed boundary sensors, missing optional package, nonfinite statistics, partial chains and recomputation after extension. Assert all physical nodes/sensors are monitored, and direct package calls reproduce every reported statistic and quantile MCSE. Plotting imports are not required to compute diagnostics.
- [ ] **Step 6: Commit the exact Task 11 files:** `git commit -m "Add physical posterior diagnostics and qualification status"`.

### Task 12: Exercise the ordinary notebook sequence and document its meaning

**Files:** Create `notebooks/HMCSampling.ipynb`, `docs/hmc.md`, `docs/api/heat_inference.md`, `tests/test_hmc_workflow.py`, `.github/workflows/hmc-cpu.yml`; modify `README.md`, `docs/index.md`, `docs/installation.md`, `docs/testing.md`, `docs/api/index.md`, `docs/conf.py`, `pyproject.toml`.

**Interfaces:**
- Consumes: the public calling convention, physical result methods and diagnostic status from Tasks 5–11.
- Produces: one executable notebook using those interfaces, an automated small execution of its computation cells, API/reference documentation, and a separate CPU sampling test job. The notebook has one tagged `parameters` cell; CI overrides it with `nmesh=4, chains=1, warmup=8, draws=12`, a temporary output and disabled plotting. This is explicitly an execution check.

- [ ] **Step 1: Write the failing notebook-sequence test.** The notebook's data are made on a finer mesh with fixed interior sensors, then supplied to a coarser inference model. Run the same computation cells using `nbclient`; use the existing notebook toolchain or add `nbformat`, `nbclient`, and `ipykernel` to development extras. Plotting cells are separately tagged and are the only skipped cells in this check.

```python
def test_documented_sequence_preserves_fields_and_saved_results(tmp_path):
    problem = make_problem(nmesh=2)
    before = state_arrays(problem.forward)
    estimate = problem.optimize(initial_h=1.0)
    samples = problem.sample(initial=estimate, seed=7, chains=1,
                             warmup=8, draws=12, output=tmp_path/"example")
    assert samples.load_draws().shape == (1, 12, 9)
    assert samples.summary().mean.shape == (9,)
    assert samples.diagnostics().qualified is False
    assert all(np.array_equal(a, b) for a, b in zip(before, state_arrays(problem.forward)))
    loaded = HeatInverseProblem.load(samples.output)
    assert loaded.to_record().problem_id == problem.to_record().problem_id
    assert loaded.resume(samples.output).metadata["draws_done"] == [12]
```

- [ ] **Step 2: Run `python -m pytest tests/test_hmc_workflow.py -q`.** Expected: the notebook/documented workflow is absent or does not yet execute; existing public methods remain covered by earlier tasks.
- [ ] **Step 3: Write the scientist-facing example and documentation.** Start from heat physics, fixed measurements, prior parameters and what the inferred field means. Show scalar/callable/nodal inputs, full-state optimization handoff, default four chains, explicit single-chain checks, dispersed/explicit starts, trajectory controls, physical plotting, sensor residuals, interruption/resume/extension, and explicit application. Explain known-noise units, log-field mode versus physical summaries, pointwise intervals versus simultaneous bands, and `T(mean(h))` versus mean temperature prediction. Include save locations, disk-backed access, optional diagnostics, CPU/one-rank limits and preserved legacy behavior. Do not present a short chain or a successful notebook as converged evidence; link Task 14's eventual scientific report.
- [ ] **Step 4: Add isolated sampling CI and lazy-import checks.** Build from `environment.yml`, install the sampling/diagnostics extras with the qualified constraints, set CPU float64 and one thread, then run the HMC tests. Mark full studies `hmc_study` and exclude them from ordinary CI. Keep the existing legacy job working when optional sampling/diagnostics packages are absent; tests requiring them may skip only in that job, never in the dedicated sampling job. Mock optional modules appropriately for the existing lightweight Sphinx build. Include source/constraints changes in workflow path filters.
- [ ] **Step 5: Run `python -m pytest -q -m "not hmc_study"` and `sphinx-build -W --keep-going -b html docs docs/_build/html`.** Expected: all required tests and docs pass in the qualified environment; separately verify legacy imports/tests in the original environment. Execute the notebook's computation cells and inspect its rendered physical plots. The repository documents a PyVista/VTK conflict with the current FEM environment: preserve that environment and use a separate plotting environment with exported NumPy fields or a lightweight Matplotlib renderer. Validate the existing-plot DOF mapping independently; report any unqualified optional plotting environment explicitly. Put Matplotlib in a separate optional `validation` extra for portable scientific figures, never a base dependency. Record actual commands, versions and outcomes; retain only small illustrative notebook outputs and keep long chains in run directories.
- [ ] **Step 6: Commit the exact Task 12 files:** `git commit -m "Document and test the conductivity posterior workflow"`.

### Task 13: Qualify numerical settings and calibrate the discrete posterior

**Files:** Create `src/hmc_validation.py`, `scripts/run_hmc_study.py`, `validation/hmc_reference.json`, `tests/test_hmc_validation.py`, `docs/validation/hmc-numerics.md`; modify `pyproject.toml`.

**Interfaces:**
- Consumes: the complete public inference workflow and Task 11 diagnostics, plus exact small-target tests from Task 7.
- Produces: `StudyConfig.from_json(path) -> StudyConfig`; `run_numerical_pilot(config, output, *, data_path: PathLike | None=None) -> dict`; `run_sbc(config, output, *, replicates=100) -> dict`; `sbc_ranks(truth, posterior_values, *, seed) -> ndarray`; `sbc_uniform_envelope(*, replicates, posterior_count, quantities, seed) -> dict`.
- CLI: `python scripts/run_hmc_study.py pilot --config validation/hmc_reference.json --output runs/hmc-pilot`, with optional `--data PATH`; `... sbc --config validation/hmc_reference.json --output runs/hmc-sbc --replicates 100`. Existing outputs resume their recorded experiment; changed configurations require a new output.
- The reference configuration fixes quadrilateral elements, source 1, bottom temperature 300 K, inference meshes `(16,32,64)`, sensors `(i/17,j/17)` for `i,j=1,...,16`, `sigma=1e-3 K`, `mu=log(2)`, `kappa=2`, `tau=1`, `h_ref=1` in the declared conductivity units, and master seed 20261026. These are recorded demonstration choices, not inferred or universal material parameters.

- [ ] **Step 1: Write failing validation-record and calibration-helper tests.** Use an exact scalar Gaussian posterior to verify rank direction, randomized handling of exact ties, and deliberate bias detection; deterministic hand-constructed ranks verify the uncertainty-envelope calculation. Unit tests use small replica counts for mechanics only and cannot certify the required 100-experiment heat study.

```python
def test_rank_counts_draws_below_truth():
    draws = np.array([[1., 3., 4.], [0., 2., 5.]])
    ranks = sbc_ranks(np.array([2., 6.]), draws, seed=8)
    assert np.array_equal(ranks, [1, 3])

def test_reference_experiment_has_fixed_physical_observations():
    config = StudyConfig.from_json("validation/hmc_reference.json")
    assert config.sensor_points.shape == (256, 2)
    assert np.all((config.sensor_points > 0) & (config.sensor_points < 1))
    assert config.sigma == 1e-3
    assert config.inference_meshes == (16, 32, 64)
```

- [ ] **Step 2: Run `python -m pytest tests/test_hmc_validation.py -q`.** Expected: missing study configuration/helpers.
- [ ] **Step 3: Implement and run the numerical pilot.** Before Task 14 exists, use the analytic uniform-conductivity solution `T=300+(y-0.5*y**2)/2` (`h=2,q=1`) at the fixed sensors, plus one noise realization with seed 20261024. Save/reuse these data across meshes; `--data` instead uses an existing observation record. At `nmesh=16`, compare fixed `(rtol,atol)` pairs `(1e-8,1e-10)`, `(1e-10,1e-12)`, `(1e-12,1e-14)` against a direct-LU reference on identical fields. Check at the estimate and at 10 seeded prior-correlated perturbations, then repeat on 10 pilot posterior states. Require `abs(delta U)<1e-4` and `norm(delta grad_U)/max(1,norm(grad_U_ref))<1e-5`; separately apply the same force criterion to `grad_phi`. Repeat at 32 and 64 before their studies. Select the loosest setting meeting all checks, record it, and freeze it for each production run. If none meet them, investigate the solves/force before proceeding.
- [ ] **Step 4: Measure initialization and sampling cost.** Run four-chain pilots at 16 with `s=0.1,0.3,1.0`, warmup 256 and 256 retained iterations, with fixed prior/data and independent recorded streams. Record initial misfits, acceptance, energy errors, step sizes, actual durations, solve counts and elapsed time. Choose the largest tested spread with finite starts, no failed PDE, no divergences in the final 64 warmup or 256 retained iterations, no step size at the lower bound, and mean retained acceptance at least 0.65 in every chain; if none qualifies, investigate and rerun a declared pilot. Use its observed ESS and timing to set an initial production length, beginning with warmup 1000 and at least 2000 retained iterations. Estimate additional length as `ceil(current_draws*400/min_ESS)` with a factor-two margin; confirm through actual diagnostics. Pilot lengths and estimated costs are not convergence evidence. Update the documented default spread only from this evidence.
- [ ] **Step 5: Implement and run simulation-based calibration on `nmesh=16`.** Generate 100 independent truths and noisy data sets from exactly the same discrete prior/heat model used for inference; use independent truth, noise and sampling seeds. Monitor area-weighted mean `h` and the four nodes `(0.25,0.25)`, `(0.25,0.75)`, `(0.75,0.25)`, `(0.75,0.75)`. Every posterior run starts optimization from the prior median field, uses four chains and the ordinary qualification rules, and never uses its generated truth as an initial estimate. Following [Talts et al., Section 5.1 and Algorithm 2](https://arxiv.org/html/1804.06788v2), choose a conservative rank-only subsampling interval using the worst quantile-indicator ESS across these quantities; account for antithetic correlations. Keep 100 approximately independent values for each rank, balanced over the chains; extend any run that lacks enough. Preserve every original chain draw and report residual autocorrelation and sensitivity to doubling this interval.
- [ ] **Step 6: Assess finite-experiment calibration uncertainty.** Show rank histograms and empirical-CDF deviations from discrete uniform ranks `0,...,100`; simulate 100,000 uniform-rank ensembles with seed 20261027 for 95% simultaneous CDF envelopes, using a Bonferroni allocation over the five reported quantities. Save individual ranks, selected indices, convergence results and failures; do not discard difficult prior draws. An envelope violation triggers investigation, not a rerun-until-pass rule. State the limited sensitivity of 100 replications; absence of a detected discrepancy is conditional evidence, not proof of exact sampling.
- [ ] **Step 7: Verify and document the numerical evidence.** Run the Task 13 unit tests, the `pilot` command and the full `sbc --replicates 100` command. `docs/validation/hmc-numerics.md` records posterior identity, actual source/environment, commands, comparisons, figures, measured costs and outcomes. Missing/failed/unqualified replications remain visible and prevent a claim that calibration is complete. Unit tests alone cannot complete this task. If candidate package/solver settings change during qualification, repeat the affected force/dynamics/continuation checks and use a new numerical identity.
- [ ] **Step 8: Commit the exact Task 13 source/configuration/report files:** `git commit -m "Qualify HMC numerics and discrete posterior calibration"`. Keep raw chains outside Git and link their immutable manifests/checksums from the report.

### Task 14: Compare smooth and sharp conductivity truths across all agreed meshes

**Files:** Extend `src/hmc_validation.py`, `scripts/run_hmc_study.py`, `validation/hmc_reference.json`, `tests/test_hmc_validation.py`, `notebooks/HMCSampling.ipynb`; create `docs/validation/hmc-mesh-study.md`; modify `docs/hmc.md`, `docs/index.md`, `docs/testing.md`.

**Interfaces:**
- Consumes: qualified numerical settings, fixed reference observations/prior and the ordinary sampling/results methods. Read `_sharp_interface_h` from `src/benchmark_utils.py`.
- Produces: `make_truth_data(config, *, truth, output) -> Path`; `run_mesh_study(config, *, data_path, nmesh, output) -> dict`; `interface_coverage(coordinates, lower, upper, truth_values, *, width=0.05) -> dict`; `write_study_report(manifests, destination) -> None`.
- CLI adds `truth --case smooth|sharp`, `mesh --data PATH --nmesh 16|32|64`, and `report --manifest PATH [PATH ...] --output PATH`; all use `--config`/`--output` where a calculation is created. Truth records contain exact sensor order, one noise realization, noiseless temperatures, refinement evidence and truth definition.

- [ ] **Step 1: Add failing tests of shared observations and interface-region reporting.** Use tiny meshes only to test mechanics; ensure every inference mesh receives byte-identical data/sigma/points from one truth record and physically identical prior parameters. Interface distance is `abs(x+0.25*y-0.62)/sqrt(1+0.25**2)`; the near region has distance `<=0.05` in unit-square length units.

```python
def test_interface_coverage_reports_regions_separately():
    xy = np.array([[0.62, 0.0], [0.1, 0.0], [0.9, 0.0]])
    report = interface_coverage(xy, np.array([0.5, 0.5, 2.5]),
                                np.array([1.5, 1.5, 3.5]),
                                np.array([3., 1., 3.]))
    assert report["near_count"] == 1
    assert report["near_coverage"] == 0.0
    assert report["away_count"] == 2
    assert report["away_coverage"] == 1.0
```

- [ ] **Step 2: Run `python -m pytest tests/test_hmc_validation.py -q`.** Expected: new truth/study/coverage interfaces fail while earlier qualification tests pass.
- [ ] **Step 3: Generate and freeze each fine-mesh observation set.** Use smooth `h=1+0.35*x+0.25*y+0.15*x*y` and the existing sharp interface. Begin reference comparisons at `nmesh=128,256`; require the maximum sensor-temperature change to be below `0.1*sigma` before generating data on the finer mesh. If this fails, refine again and record every comparison; unresolved truth discretization prevents claiming a qualified reference. Use seeds 20261026 and 20261027 for the two noise realizations. Save once and reuse across 16/32/64; never generate fresh noise per mesh. This fine-truth test remains separate from same-mesh calibration.
- [ ] **Step 4: Run four-chain smooth and sharp studies at 16, 32 and 64.** First qualify the force tolerance using Task 13's `pilot --data` on each actual observation set and mesh; use the same physical prior. Choose run lengths from pilot cost/ESS, extend explicitly as needed, and report all six outcomes. No runtime cap or monotonic acceptance condition is imposed. If persistent divergences/failed convergence demand altered tuning, retain the original run and start a labelled new calculation with fresh warmup; never retune retained draws or hide the unsuccessful case. An unresolved scientific failure remains an open qualification item even if the execution tests pass.
- [ ] **Step 5: Quantify accuracy, exploration and cost separately.** Report all-node diagnostics and worst locations, mean/quantile MCSE, acceptance/energy distributions, autocorrelation, sensor residuals, total/retained wall time, heat/adjoint solve counts, and worst/representative ESS per second and per solve. Give uncertainty for acceptance and ESS/cost comparisons using 500 seeded resamples of contiguous chain blocks, with block length at least twice the largest measured correlation time. If too few blocks remain, report that the uncertainty estimate is unresolved. Compare physical summaries at shared locations across meshes with Monte Carlo errors shown; do not label discretization differences as sampling errors. Fixed-sensor predictive residuals include observation noise when judging predictive adequacy.
- [ ] **Step 6: Report the Gaussian prior's limitations near an interface.** Compute nodal 90% interval coverage separately inside/outside the fixed physical band, plot truth/mean/interval width and residuals, and state that this is a spatial diagnostic from one field/noise realization. Perfect interface recovery and nominal 90% spatial coverage are not correctness criteria. Repeated-noise coverage claims require a separate repeated experiment. Keep the baseline prior fixed. Add predeclared one-at-a-time sensitivity calculations at `nmesh=16` for each truth with `kappa=1,4` at baseline `tau=1`, and `tau=0.5,2` at baseline `kappa=2`; retain `mu,h_ref` and the same data. Report all eight additional outcomes, including failures, and their effects on means/intervals/predictions. These are separate posteriors and cannot be selected to force truth recovery. Explain that the continuous nodal conductivity represents a jump through transition cells; a future exactly discontinuous model may need a different field representation as well as a different prior.
- [ ] **Step 7: Verify the report and final integration.** Run `python -m pytest -q -m "not hmc_study"`, the truth/mesh/report commands for both cases and all three meshes, then the warning-free Sphinx build. Inspect the notebook's physical plots and the report's scientific figures. Record actual manifests/checksums and leave any unmet convergence/calibration/continuum interpretation requirements explicitly open. A short run or an absent study is not a completed Task 14.
- [ ] **Step 8: Commit the exact Task 14 report and source files:** `git commit -m "Validate conductivity posteriors across meshes and interface truths"`. Follow the chosen execution method's whole-branch review before declaring implementation complete.

## Requirement coverage and completion evidence

| Accepted requirement | Owning tasks / evidence |
| --- | --- |
| Q1 shared posterior; Q2 existing behavior; Q3 staged validation | 4–5; 1/10/12 legacy regression; 6–7/13–14 numerical evidence |
| Q4 complete-state handoff; Q5 fixed measurements | 3/5/8–9; 2/4/14 observation identity |
| Q6 future sharp-field priors; Q7 meshes 16/32/64 | 3/5 capability checks and complete state; 13–14 |
| Q8 physical draws, predictions and plots; Q9 caller preservation | 10–12; 1/4/5/9/10 success/failure/interruption tests |
| Q10 explicit hyperparameters and sensitivity; Q11 smooth/sharp demonstrations | 3/8/13 reference record and 14 separate sensitivity runs; 13–14 |
| Q12 log-conductivity mode and full-field initialization; Q14 dynamics/warmup/controls | 1/3/5/9 mode and initialization checks; 6–7 Hamiltonian, known-target and failure tests |
| Q13 automatic saves/continuation; Q15 diagnostics | 8–9 abrupt-death/exact-resume tests; 11/13–14 |
| Q16 fixed fine-mesh observation experiment; Q17 BlackJAX | 2/13–14 fixed 256 sensors and shared fine-mesh data; 6–7 direct-versus-connected and library qualification |
| Q18 per-chain counts/seeds; Q19 frozen experiment | 8–9; 1/2/5/8 mutation and reconstruction tests |
| Q20 dispersed starts; Q21 disk-backed physical summaries | 9/13 sensitivity pilot; 8/10–11 |
| Q22 four-chain default; Q23 resume versus extension | 9/11; 8–9 uneven-count and extension tests |
| Q24 matching destination before application | 1/10 fingerprints, constant promotion, rollback and stale-form tests |

A completed implementation has passing numerical, integration and legacy checks; a reproducible ordinary notebook sequence; and saved scientific-study evidence. Reports must distinguish numerical correctness, chain convergence, mesh sensitivity and prior adequacy. Unmet scientific criteria remain visible even when the software executes successfully.

## Primary implementation references

These references informed implementation choices on 2026-10-26; check the pinned installed source during execution. Documentation is not local qualification.

- [JAX supplied reverse derivatives](https://docs.jax.dev/en/latest/_autosummary/jax.custom_vjp.html) and [scoped eager execution](https://docs.jax.dev/en/latest/_autosummary/jax.disable_jit.html) support the proposed connection. [Pure callback semantics](https://docs.jax.dev/en/latest/_autosummary/jax.pure_callback.html) motivate keeping failure-producing solves in the eager host sequence.
- [BlackJAX HMC source](https://blackjax-devs.github.io/blackjax/_modules/blackjax/mcmc/hmc.html), [integrator state/source](https://blackjax-devs.github.io/blackjax/_modules/blackjax/mcmc/integrators.html), and [step-size adaptation](https://blackjax-devs.github.io/blackjax/autoapi/blackjax/adaptation/step_size/index.html) define the interfaces exercised in Tasks 6–7.
- [BlackJAX 1.7.1 requirements](https://raw.githubusercontent.com/blackjax-devs/blackjax/1.7.1/pyproject.toml), [JAX 0.11.2](https://pypi.org/project/jax/0.11.2/), [JAXlib 0.11.2 distributions](https://pypi.org/project/jaxlib/0.11.2/), and [ArviZ Stats 1.3.3](https://pypi.org/project/arviz-stats/1.3.3/) support the candidate environment choices.
- [ArviZ Stats R-hat](https://python.arviz.org/projects/stats/en/latest/api/generated/arviz_stats.rhat.html), [ESS](https://python.arviz.org/projects/stats/en/latest/api/generated/arviz_stats.ess.html), and [MCSE](https://python.arviz.org/projects/stats/en/latest/api/generated/arviz_stats.mcse.html) supply established diagnostic calculations, with project-specific qualification rules in Task 11.

## Plan review record

Self-review completed on 2026-10-26 against the consolidated specification and both interview records. All Q1–Q24 requirements have owning tasks. Checked dependency order, public types, physical and native DOF ordering, optional dependencies, all five Review Focus cases, and explicit scientific completion evidence. Corrected the Q12/Q16 mapping, retained-state prediction cache on resume, complete-state capability checks, and explicit application to a constant-valued caller. Test examples are syntax-checked; relative specification links and task numbering are checked. The plan is comparable in length to the specification and contains decisions/tests rather than implementation bodies.

Status: ready for the user's plan review. No numerical implementation, dependency installation or new sampling result is claimed. Candidate JAX/BlackJAX compatibility and scientific qualification remain work to be executed. The execution method is still open; the writing-plans skill requires review and a choice before implementation.
