---
orphan: true
---

# Saved conductivity fields and continuation

The saved history contains physical conductivities and predicted thermometer
temperatures at every retained iteration. Inspecting it does not require another
inference calculation. The examples below continue the [complete short
experiment](hmc.md#a-complete-deliberately-short-experiment): `output`, `forward`
and `probabilities` are its variables. In a later Python session, set `output` to
the printed directory and define
`probabilities = (0.025, 0.16, 0.5, 0.84, 0.975)`.

## Read physical arrays

```python
from posterior_samples import PosteriorSamples

with PosteriorSamples.open(output) as saved:
    xy = saved.coordinates.copy()
    cells = saved.cells.copy()
    h_one = saved.field(chain=0, draw=0)
    block = saved.load_draws(chains=[0], start=0, stop=10)
    predicted_kelvin = saved.sensor_predictions(chain=0)
    summary = saved.summary(quantiles=probabilities)
    print(summary.counts, summary.qualified)
```

`h_one[i]` belongs to `xy[i]`, and each row of `cells` indexes those same nodes.
`block` has axes `(chain, retained draw, node)`; `predicted_kelvin` has axes
`(retained draw, thermometer)`, in the original observation order. Chain and draw
indices start at zero and `stop` is exclusive. These labels avoid silently
permuting a conductivity field when the finite-element numbering changes.
Keep physical arrays with their coordinates and connectivity when exporting them.

These operations require NumPy and the standard-library archive reader, without
MPI, JAX, BlackJAX, ArviZ Stats or plotting libraries. `field` returns a copy;
spatial labels and summary arrays are immutable. `load_draws` explicitly allocates
the selected history. `summary` reads bounded chunks of nodes but includes every
available retained state. It uses the saved convergence report if it still
matches; it does not recompute diagnostics. All rejection repeats count as
retained states. Removing repeated values would change the posterior measure.

Interrupted chains may have unequal retained lengths. Their pooled summary gives
longer histories proportionally more weight and cannot qualify an unfinished
calculation. `load_draws()` refuses unequal lengths rather than silently truncating:
choose explicit chains and a common `start`/`stop` known to exist in each.
With no retained draws, there is no physical summary to report.

## Reassess convergence at the intervals being used

```python
with PosteriorSamples.open(output) as saved:
    report = saved.diagnostics(quantiles=probabilities)
    summary = saved.summary(quantiles=probabilities)
    print(report.status, report.reasons, report.persisted)
    print(summary.counts, summary.qualified)
```

Diagnostics optionally import the `[diagnostics]` extra and write the advisory
`diagnostics.json` report; they require neither a heat solve nor JAX. If that
extra is absent, the report is unavailable and unqualified. A report for an older
history or different unsupported probabilities cannot qualify the current
summary. Inspect report status and persistence as well as the numerical thresholds
in the [tutorial](hmc.md#physical-uncertainty-and-qualification). In this short
one-chain example the result remains unqualified before and after continuation.

## Reconstruct temperatures and finite-element functions

These optional operations require the native finite-element/MPI environment.
They do not require a new sampling calculation:

```python
with PosteriorSamples.open(output) as saved:
    T_one = saved.temperature(chain=0, draw=0)
    h_function = saved.as_function(chain=0, draw=0)
    mean_function = saved.as_function(summary="mean")
    median_function = saved.as_function(summary=("quantile", 0.5))
```

`temperature` performs a private heat solve and returns temperatures in the
recorded canonical node order, without changing the caller's fields.
`as_function` builds a private mesh and conductivity function without a heat
solve. Its finite-element order is suitable for the existing plotting routines;
do not assume it equals the archive's canonical node order. Always use its own
function space for a plot. A separate plotting environment can instead read saved
arrays with their spatial labels. Optional Matplotlib is in `[validation]`;
PyVista/VTK `[plot]` needs separate native-library qualification, as described in
[installation](installation.md#measured-macos-compiler-setup).

Consume or export these functions with their own mesh labels. Ordinary copied
NumPy arrays can be inspected after the saved-result context closes. The reviewed
resource-lifetime defect concerns a different object: a completed chain can leave
its private adjoint field and compiled form reachable until the enclosing inverse
problem closes. Close the enclosing problem when finished, particularly between
repeated continuation calculations; result-handle closure alone does not resolve
that accumulation.

## Explicitly change the caller's physical state

Optimization, summaries and private predictions leave the caller's conductivity
and temperature unchanged. To choose a field for a further heat calculation,
explicitly request its application:

```python
with PosteriorSamples.open(output) as saved:
    T_at_mean_h = saved.apply_to(forward, summary="mean")
    # Alternatively: saved.apply_to(forward, chain=0, draw=0)
```

This checks mesh/connectivity, heat source and boundary temperatures, updates
the supplied `forward` conductivity and solves its matching temperature. That
solver remains owned by the caller. The returned temperature is $T(E[h])$,
which generally differs from $E[T(h)]$. A standard deviation is a spread for
plotting, not an applicable physical conductivity. A destination mismatch or
failed application leaves the destination unchanged.

## Continue the same physical experiment

Use the same numerical environment and compiler settings as the original run:

```python
from heat_inference import HeatInverseProblem

with HeatInverseProblem.load(output) as restored:
    with restored.resume(output) as completed:
        print(completed.summary(quantiles=probabilities).counts)
    with restored.resume(output, total_draws=16) as extended:
        report = extended.diagnostics(quantiles=probabilities)
        summary = extended.summary(quantiles=probabilities)
        print(summary.counts, report.status, summary.qualified)
```

`resume(output)` finishes the originally requested retained count after an
interruption. `total_draws=16` means **sixteen retained iterations in total per
chain**, not sixteen additional iterations. For a longer calculation choose
`total_draws=N` with `N` above the current target. Extension invalidates the old
qualification before another draw is written. Recompute current diagnostics
with the desired probabilities when it finishes. Partial histories remain
visible if the calculation stops again.

The loaded problem owns reconstructed private heat/adjoint workspaces; it does
not acquire the caller's `forward`. Its context manager and each result context
close their owned resources. Every committed trajectory retains its random
state, adaptation and continuation state; a restart recovers the last committed
trajectory rather than inventing replacement samples. Keep heat source,
boundaries, mesh, sensor positions, measured values/errors, prior, numerical
tolerances and recorded environment unchanged. A different experiment needs a
new output directory.

The measured macOS setup also records `FI_PROVIDER`, `FI_TCP_IFACE` and
`FI_SOCKETS_IFACE`, including unset values: changing them prevents append. The
review nevertheless found that effective local/user DOLFINx and FFCx settings
were omitted from numerical identity checks. **Matching recorded identities
alone do not establish unchanged form-compiler settings.** Keep those settings
unchanged independently; the [dated numerical evidence](validation/hmc-status.md#numerical-reproducibility)
records this open defect. Archive inspection remains possible without the
original MPI environment.

## Preserve the history

A new sampling output directory contains `problem.json`, `problem.npz`,
`chains.sqlite` and advisory `status.json`. Diagnostics add `diagnostics.json`.
SQLite records warmup and retained iterations transactionally. Copy a **closed**
run directory or use SQLite's backup operation; copying only an active database
does not preserve a reliable backup. Keep long histories outside notebook output
cells and version-controlled documentation. Advisory status is not a substitute
for the committed history or a fresh diagnostic report.

The [16 November review limits](validation/hmc-status.md#numerical-reproducibility)
also include continued admission following pilot-memory or SQLite-storage
exhaustion. A failed calculation must be investigated; this guide does not treat
those failure modes as repaired or authorize changes to an existing scientific
archive.

## Recovery evidence and unfinished extensions

Transactional history preservation does not guarantee safe admission of another
calculation after resource exhaustion. The [validation guide](testing.md#open-recovery-and-notebook-limitations)
separates existing continuation checks from the still-open pilot-memory,
SQLite-storage, compiler-identity and closed-chain-memory findings. After an
exhaustion failure, preserve the committed archive, inspect the cause and restore
capacity before starting more work. No repair of these findings is claimed here.

For mesh/prior studies, an extension request and an older `result.json` can
coexist. The shared study report labels that situation `extension_in_progress`;
the final notebook cell currently reads the old result directly and can display
old figures as if current. It also searches `../runs`, which selects the wrong
location when launched at the project root. The dated [notebook checks](testing.md#open-recovery-and-notebook-limitations)
reproduce both defects. Avoid that final cell on scientific archives until its
separate correction: select the intended study explicitly, inspect the
shared generation-aware report, and label any previous figures as historical.
An old qualified posterior does not qualify an unfinished extension.
