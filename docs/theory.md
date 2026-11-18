# Theory

## Forward problem

On the unit square $\Omega = [0,1]^2$, the steady-state heat equation is

$$-\nabla\cdot\bigl(h(x,y)\,\nabla T\bigr) = q(x,y),$$

with a Dirichlet condition on the bottom wall and insulated (zero-flux Neumann) conditions
on the other three walls:

$$T(x, y{=}0) = T_0, \qquad \nabla T\cdot\hat{n} = 0 \ \text{ on } \ \partial\Omega\setminus\{y=0\}.$$

$T_0$ defaults to $300\,\mathrm{K}$ (`DBC_value`). The zero-flux Neumann condition is
*natural* — it needs no explicit term in the weak form.

Discretizing with Q1 elements on quadrilaterals or P1 elements on triangles gives the
weak form implemented in {py:class}`forward_solver.SteadyHeat2DForwardSolver`:

$$a(u,v) = \int_\Omega h\,\nabla u\cdot\nabla v \ \mathrm{d}x,
\qquad
L(v) = \int_\Omega q\,v \ \mathrm{d}x.$$

The resulting system is symmetric positive definite (guaranteed by $h > 0$), which is why
the default KSP configuration is conjugate gradient with a `hypre` preconditioner.

The following legacy reconstruction sections describe the original TAO calculation.
The [fixed-sensor posterior](#fixed-sensor-conductivity-posterior) below has a
separate probability law; the legacy TAO calls retain their original objective.

## Legacy nodal observation model

A synthetic measurement is produced by adding uncorrelated Gaussian noise to the forward
solution:

$$T_{\rm obs}(x,y) = T(x,y) + \varepsilon, \qquad \varepsilon \sim \mathcal{N}(\mu, \sigma^2).$$

See {py:meth}`forward_solver.SteadyHeat2DForwardSolver.add_noise`. The noise is drawn on
rank 0 and broadcast, so the observation is identical across MPI ranks.

## Legacy reconstruction objective

Recover $h$ by minimizing the regularized misfit

$$J[T(h), h] = \tfrac{1}{2}\left[\int_\Omega \frac{\bigl(T(h) - T_{\rm obs}\bigr)^2}{\sigma^2}\ \mathrm{d}x
\; + \; \alpha \int_\Omega \lvert\nabla h\rvert^2 \ \mathrm{d}x \right]$$

subject to the PDE constraint and the positivity bound $h \ge h_{\min}$.

The second term is **Tikhonov ($H^1$) regularization**: it penalizes roughness in $h$ and
is what makes the otherwise ill-posed reconstruction stable against sensor noise. Multiplying the objective by $\sigma^2$ shows that $\alpha\sigma^2$ sets the relative
weight of roughness against the unweighted squared residual. Too little smoothing
can imprint noise onto $h$; too much can suppress genuine structure.

## Legacy discrete adjoint gradient

Introduce the Lagrangian with multiplier (adjoint state) $\lambda$,

$$\mathcal{L} = J - \langle\lambda, g[T(h), h]\rangle,
\qquad g[T(h),h] = -\nabla\cdot(h\nabla T) - q,$$

Requiring stationarity with respect to $T$ gives the **adjoint equation**

$$-\nabla\cdot(h\nabla\lambda) = \frac{T - T_{\rm obs}}{\sigma^2},
\qquad \lambda(x, y{=}0) = 0,$$

with zero-flux Neumann conditions on the remaining walls. In weak form
({py:class}`adjoint_solver.SteadyHeat2DAdjointSolver`):

$$a(u,v) = \int_\Omega h\,\nabla u\cdot\nabla v \ \mathrm{d}x,
\qquad
L(v) = \frac{1}{\sigma^2}\int_\Omega (T - T_{\rm obs})\,v \ \mathrm{d}x.$$

:::{important}
The adjoint left-hand side is **identical** to the forward one — the operator
$\nabla\cdot(h\nabla\cdot)$ is self-adjoint. One gradient therefore costs roughly two
linear solves, *independently of the number of degrees of freedom*. This is the whole
point of the adjoint method: a finite-difference gradient would cost one solve per DOF.
:::

The total derivative, assembled by
{py:meth}`adjoint_solver.SteadyHeat2DAdjointSolver.assemble_gradient` against a test
function $v$, is

$$\frac{\mathrm{d}J}{\mathrm{d}h}[v] = \int_\Omega \Bigl(-\nabla T\cdot\nabla\lambda\, v
\; + \; \alpha\,\nabla h\cdot\nabla v\Bigr)\ \mathrm{d}x.$$

### Adjoint Dirichlet value

The adjoint boundary condition should be $\lambda(y{=}0) = 0$, and `DBC_value` defaults to
`0.0` accordingly. Passing a non-zero constant instead is currently *harmless*: with a
single constant-Dirichlet edge and Neumann conditions elsewhere, a constant field lies in
the kernel of $\nabla\cdot(h\nabla\cdot)$, so $\lambda_c = \lambda_0 + c$ and the gradient
— which only ever sees $\nabla\lambda$ — is unchanged. It would silently break if the
Dirichlet data became non-constant or a second Dirichlet edge were introduced. Prefer
`DBC_value=0.0`.

## Legacy log parametrization

By default ({py:class}`tao_solver.SteadyHeat2DTAOSolver`, `use_logh=True`) the
optimization variable is

$$m = \log h,$$

which enforces positivity structurally and conditions the problem better when $h$ spans
orders of magnitude. The chain rule is applied to the gradient,

$$\frac{\mathrm{d}J}{\mathrm{d}m} = \frac{\mathrm{d}J}{\mathrm{d}h}\,\frac{\mathrm{d}h}{\mathrm{d}m}
= h\,\frac{\mathrm{d}J}{\mathrm{d}h},$$

and the bounds are mapped to $[\log h_{\min}, \log h_{\max}]$. With `use_logh=False` the
optimization runs directly in $h$ with bounds $[h_{\min}, h_{\max}]$.

## Legacy TAO optimization loop

Each TAO iteration evaluates the objective-and-gradient callback
`SteadyHeat2DTAOSolver._objgrad`:

1. Write the trial vector $X$ back into the **shared** `fwd.h.function` (exponentiating
   first if optimizing in $\log h$). Both the forward and adjoint UFL forms hold this
   function *by handle*, so both see the update.
2. Forward solve $\Rightarrow T$.
3. Assemble the misfit and regularization scalars $\Rightarrow J$.
4. Adjoint solve $\Rightarrow \lambda$.
5. Assemble $\mathrm{d}J/\mathrm{d}h$, apply the chain rule if needed, return $(J, G)$.

The default TAO type is `blmvm` (bound-constrained limited-memory variable metric), which
handles simple bounds and requires only the gradient — no Hessian.


## Fixed-sensor conductivity posterior

Here $N$ is the number of conductivity nodes and $S$ the number of fixed physical
thermometers. The spatial conductivity uses all P1/Q1 nodal coefficients $h_i$.
The data $d_s$ and known positive errors $\sigma_s$ are temperatures in kelvin.
The interpolation matrix $B$ predicts temperatures at the observation positions;
its rows remain associated with those positions when the inference mesh changes.
Independent Gaussian errors give

$$
d_s=[BT(h)]_s+\epsilon_s,\qquad
\epsilon_s\sim\mathcal N(0,\sigma_s^2),\qquad
\Sigma=\operatorname{diag}(\sigma_s^2),
$$

$$
\Phi(h)=\tfrac12(BT(h)-d)^T\Sigma^{-1}(BT(h)-d),\qquad
p(d\mid h)\propto e^{-\Phi(h)}.
$$

There is **no finite-element mass matrix in the sensor likelihood**. These are
individual thermometer readings, whereas the legacy objective integrates a
spatial temperature residual. Likewise the prior below is on log conductivity,
rather than the legacy roughness penalty on physical conductivity.

### Gaussian field and whitened coordinates

For nodal basis functions $\varphi_i$, define

$$
\begin{aligned}
M_{ij}&=\int_\Omega\varphi_i\varphi_j\,dx,&
K_{ij}&=\int_\Omega\nabla\varphi_i\cdot\nabla\varphi_j\,dx,\\
M_L&=\operatorname{diag}(M\mathbf1),& A&=\kappa^2M+K.
\end{aligned}
$$

The prior map is

$$
\begin{aligned}
m&=\log(h/h_{\rm ref})=\mu+Lz,&
L&=\tau^{-1}A^{-1}M_L^{1/2},\\
z&\sim\mathcal N(0,I),&h_i&=h_{\rm ref}\exp(m_i).
\end{aligned}
$$

The exponential acts on the nodal values; the conductivity in the heat equation
is then their P1/Q1 interpolant. The covariance of the nodal log field is
$LL^T=\tau^{-2}A^{-1}M_LA^{-T}$. Natural homogeneous Neumann conditions apply to
the prior on **every** wall, including the wall with imposed temperature. The
conductivity prior does not inherit the temperature Dirichlet condition.
Positive $\kappa$ makes the constant prior mode proper; positive $\tau$ controls
amplitude. Every conductivity node remains an inferred degree of freedom.

$m$, $\mu$ and $z$ are dimensionless; $h_{\rm ref}$ has conductivity units and
$\kappa$ has inverse coordinate-length units. With dimensional coordinates in
this two-dimensional white-noise construction, $\tau$ has length units so that
$Lz$ is dimensionless. On the nondimensional unit square its supplied numerical
value sets amplitude in those coordinate units. Neither $\tau^{-1}$ alone nor
$\kappa^{-1}$ is an exact nodal standard deviation or a uniquely defined
correlation length. Fix and declare all four prior parameters for an experiment.

The posterior potential used by both new inference methods is

$$
U(z)=\Phi(h(z))+\tfrac12z^Tz,\qquad p(z\mid d)\propto e^{-U(z)}.
$$

For fixed mesh and prior, $m=\mu+Lz$ is invertible with a **constant** Jacobian.
Its determinant changes only the normalization of the density in $m$.
Consequently minimizing $U$ gives the same mode in $z$ and $m$; the returned
physical field represents a mode in **log conductivity**. The density with
respect to physical nodal conductivity has a different volume element:

$$
p_h(h\mid d)=p_m(\log(h/h_{\rm ref})\mid d)\prod_{i=1}^{N}h_i^{-1}.
$$

Its mode generally differs. Adding $\sum_i\log h_i$ to $U$ would therefore
change the target being optimized or sampled. The implemented calculation adds
no such term. Transform each retained draw to $h$ to obtain physical means and
credible intervals; $E[h]$ generally differs from $h_{\rm ref}\exp(E[m])$.

### Sensor adjoint and conductivity force

Let $H(h)$ denote the heat stiffness matrix on free temperature nodes, and
$r=BT(h)-d$. The adjoint has zero values at imposed-temperature nodes and solves

$$
H(h)^T\lambda=B^T\Sigma^{-1}r
$$

on the free nodes. The constrained entries of the right-hand side are zeroed:
temperature variations vanish there, even for a thermometer on that wall.
`SensorOperator.transpose` applies the dual sensor vector directly. Inserting a
mass matrix here would describe a different likelihood. The heat operator is
symmetric, so the same assembled heat matrix supplies this adjoint solve.

Differentiating the discrete heat constraint gives the nodal derivative

$$
(g_h)_i=\frac{\partial\Phi}{\partial h_i}
=-\int_\Omega\varphi_i\,\nabla T\cdot\nabla\lambda\,dx,\qquad
(g_m)_i=h_i(g_h)_i.
$$

These are derivatives with respect to nodal coefficient vectors, without an
extra mass inverse or an $L^2$ projection. The exact transpose of the prior map
then gives the posterior gradient supplied to the Hamiltonian calculation
(the mechanical force is $-\nabla_z U$):

$$
\nabla_z\Phi=L^Tg_m
=\tau^{-1}M_L^{1/2}A^{-T}g_m,\qquad
\nabla_z U=\nabla_z\Phi+z.
$$

Matrices above share one ordering; the implementation converts between native
finite-element order and the canonical physical node order of public fields.
Internally it solves for excess temperature above a fixed boundary reference
$T_*$ and evaluates $B(T-T_*)-(d-T_*)$. This preserves the physical residual
while avoiding subtraction of rounded absolute temperatures in the force.
Public predictions retain their physical temperature origin.

One likelihood-gradient evaluation needs **one heat solve and one adjoint
solve**, independently of $N$, together with the sparse prior-map operations.
A trajectory contains many force evaluations. This is not a two-solve cost per
trajectory or retained sample. Both solves use fixed `HeatNumerics` tolerances
and zero initial guesses, independent of previous trajectories; stopping rules
do not adapt to the sampled field or to warmup. Their default values and current
qualification limits are in the [reference](api/heat_inference.md#physical-inputs)
and [numerical evidence](validation/hmc-numerics.md).

### Split Hamiltonian dynamics

Introduce an independent momentum $p\sim\mathcal N(0,I)$ and split the
Hamiltonian into a Gaussian part and a likelihood part:

$$
\mathcal H(z,p)=\underbrace{\tfrac12z^Tz+\tfrac12p^Tp}_{\mathcal H_0}
+\underbrace{\Phi(h(z))}_{\mathcal H_1}.
$$

For spacing $\varepsilon$, one reversible symmetric step first applies a
likelihood half-kick, then an exact Gaussian rotation, then a second half-kick:

$$
\begin{aligned}
p^- &= p-\tfrac{\varepsilon}{2}\nabla_z\Phi(z),\\
z'&=\cos\varepsilon\,z+\sin\varepsilon\,p^-,\\
\widetilde p&=-\sin\varepsilon\,z+\cos\varepsilon\,p^-,\\
p'&=\widetilde p-\tfrac{\varepsilon}{2}\nabla_z\Phi(z').
\end{aligned}
$$

The rotation integrates $\mathcal H_0$ exactly; the splitting still introduces
an error for the full posterior. The final Metropolis probability
$\min(1,\exp[-(\mathcal H_{\rm final}-\mathcal H_{\rm initial})])$ corrects the
finite-spacing proposal. Rejection retains the previous field and counts as
another state in the history.

The actual calls are BlackJAX `blackjax.mcmc.hmc.init` and
`blackjax.mcmc.hmc.build_kernel` with the supplied split integrator and identity
inverse momentum covariance. BlackJAX supplies momentum refresh, endpoint
proposal and Metropolis choice. Its
`blackjax.adaptation.step_size.dual_averaging_adaptation` adjusts **only step
size** during warmup. This is a custom split-Hamiltonian calculation; it does
not invoke default NUTS or an unchanged stock HMC integrator.

The existing discrete adjoint supplies the energy and derivative through an
eager custom reverse-differentiation connection (`jax.custom_vjp`). A combined
value-and-gradient request reuses that supplied derivative. The trajectory's
force call uses the same checked energy/adjoint pair directly. The heat equation
is not differentiated through its iterative solver. CPU float64 and eager
execution with `jax.disable_jit()` are required; compiled, batched and parallel
mapping transformations of this connection are unsupported.

Independent chains use separate random streams. Each trajectory draws a duration
in $(\pi/4,3\pi/4)$ by default and rounds duration/spacing to the nearest positive
integer step count; the recorded actual duration is that count times spacing.
After warmup the spacing and duration law are frozen. No momentum-covariance
adaptation occurs. The [controls reference](api/heat_inference.md#trajectory-controls)
records conditional defaults and the explicit-spacing requirement for zero
warmup.

Nonfinite proposal arithmetic is a divergent rejection with zero acceptance.
A finite absolute energy error above `divergence_threshold` is a divergence
diagnostic; it still receives the ordinary Metropolis decision. An unconverged
heat or adjoint solve at a finite conductivity is a fatal `HeatSolveFailure`,
with the stage, position, residual, reason and iteration count retained; it is
not silently replaced by a rejection. A required spacing below its recorded
lower bound also stops the calculation. Failures leave inspectable saved history.
The [dated evidence](validation/hmc-status.md#numerical-reproducibility) records
other unresolved failure and continuation limits.

For executable use see the [conductivity tutorial](hmc.md); for saved fields and
closure see the [result guide](hmc-results.md). Convergence, calibration,
finite-element accuracy and physical-prior adequacy require separate evidence.
