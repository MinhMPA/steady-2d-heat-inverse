# Context

Glossary for the steady-state 2D heat inverse problem. Terms only — no implementation
detail. When a term here conflicts with usage in code or conversation, this file wins or
gets updated; it should never be silently contradicted.

## Scope boundary

**Steady-state.** The forward model has no time dependence, and observations are single
values per sensor rather than time series. This is a deliberate boundary, not an
oversight: the classical industrial inverse heat conduction problem is transient, and
supporting it would require a backward-in-time adjoint. Treat transient work as a separate
project rather than an increment.

## The quantity being recovered

**Conductivity** — the spatially varying thermal conductivity of the domain, written `h(x,y)`
throughout this project.

> **`h` never means mesh size here.** In finite-element literature `h` universally denotes
> the mesh spacing, but this project has used it for conductivity since inception, across
> source, tests, notebooks and the published documentation. For anything to do with
> discretization scale use **level**, **nmesh**, or **mesh size** — never a bare `h`.

**Parameter vector**:
The discrete coordinates representing the unknown conductivity during optimization
or sampling. These need not be conductivity values themselves; the original
reconstruction normally uses log conductivity.

**Full-field inference**:
Inference over every conductivity degree of freedom on the chosen spatial
discretization. It does not restrict the unknown field to a selected subset of
spatial modes, and does not by itself imply a continuum-limit result.
_Avoid_: Full-resolution inference (unless the resolution is specified).

**Conductivity posterior**:
The probability law for the conductivity field conditioned on specified
temperature observations, a measurement-noise model, and a prior. Posterior
optimization and posterior sampling study this same probability law.

**Inference problem**:
The fixed heat model, temperature observations, noise law, prior, and spatial
discretization that specify a conductivity posterior. Changing any of these
declares a different inference problem.

**Original reconstruction**:
The conductivity estimate obtained by minimizing an integrated temperature
misfit plus a conductivity-roughness penalty. It is distinct from a posterior
estimate whenever the declared measurements or prior differ.
_Avoid_: Posterior mode (unless the objective is the specified posterior density).

**Initial conductivity**:
The physical conductivity field from which an inference calculation starts. It
does not specify the conductivity prior or its mean.

**Inferred state**:
The values of all unknown fields and parameters in the declared inference
problem. Conductivity alone specifies the first fixed-prior model, but a future
hierarchical model may also infer other fields or prior parameters.

**Posterior mean conductivity**:
The expectation of the physical conductivity field under its posterior. It is
not generally the exponential of the posterior mean log conductivity.

**Posterior mode in log conductivity**:
The field maximizing the posterior density expressed in log conductivity,
reported as its corresponding physical conductivity. It generally differs from
both posterior mean conductivity and the mode of the density expressed in
conductivity itself.
_Avoid_: Best-fit conductivity (without specifying the density or objective).

**Posterior mean temperature**:
The expectation of the temperature predicted by posterior conductivity fields.
It is not generally the temperature predicted at the posterior mean conductivity.

**Pointwise credible interval**:
A posterior interval for conductivity at one specified spatial location. A
collection of these intervals is not a simultaneous credible band for the field.

**Sharp conductivity variation**:
A rapid spatial change in conductivity, including a possible jump between
materials. It describes the physical field rather than a requirement that an
inferred field reproduce every transition exactly.

## Measurement

**Sensor** — a fixed physical location at which temperature is measured. Position is a
property of the instrument, not of any mesh, and does not change when the discretization
changes.

**Measurement grid**:
The fixed physical positions of the temperature observations, distinct from the
mesh used for inference. For imported nodal observations these are the original
data-node positions, retained when the inference mesh changes.

**Observation** — the vector of measured temperatures, one entry per sensor. Its length is
a property of the instrument and is therefore invariant under mesh refinement. Written `d`.

**Sensor operator** — the map from a discretized temperature field to predicted sensor
readings. Written `B`. Each mesh has its own `B`, but every `B` produces an observation of
the same length, which is what makes predictions from different meshes comparable.

**Nodal observations**:
Temperature observations at the nodes of their original data mesh, paired with
those physical coordinates. They remain the same observations if a different
mesh is subsequently used for inference.

**Observation noise**:
Measurement error modelled here as independent, zero-mean Gaussian error with
known positive standard deviations, either common to all sensors or specified
individually. This model does not include sensor-position uncertainty.

**Inverse crime** — generating synthetic observations from the same discretization used to
invert them. Flatters results, because the forward operator can reproduce data it
produced. Avoided by generating on a fine mesh and sampling at the measurement grid.

## Fidelity

**Fidelity** — unqualified, this term is ambiguous in this project and should be avoided.
There are two independent axes, and they are not interchangeable.

**Solver fidelity** — how accurately a *fixed* discrete problem is solved, controlled by the
linear solver's relative tolerance. Changing it perturbs the accuracy of evaluating the
objective; it does not change the objective, the parameter space, or their dimensions.

**Mesh fidelity** — which discrete problem is being solved, controlled by the resolution of
the discretization. Changing it changes the parameter space and its dimension, so two mesh
fidelities give genuinely *different* objectives that approximate a common continuum one.

> The asymmetry is load-bearing. Solver fidelity is inexact evaluation of one function;
> mesh fidelity is a different function. Designs that treat them as two settings of one
> knob get the transfer operators and the likelihood wrong.

**Level** — one rung of the mesh hierarchy, from coarsest to finest.

**Ladder** — the ordered collection of levels together with the operators that move
quantities between adjacent ones.

**Prolongation** — moving a quantity from a coarser level to a finer one. Correct for
*primal* quantities such as the conductivity, which are functions.

**Restriction** — moving a quantity from a finer level to a coarser one. Gradients are
*dual* quantities (linear functionals, not functions), so they restrict by the transpose of
prolongation rather than by interpolation. Using interpolation for both still converges —
to the wrong answer.

**Discretization error** — the gap between a level's discrete solution and the continuum
one. It limits continuum accuracy, while sampling a fixed discrete posterior may still
require tighter heat-solve accuracy to resolve posterior energies and forces.
