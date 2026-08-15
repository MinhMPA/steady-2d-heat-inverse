# Theory

## Forward problem

On the unit square $\Omega = [0,1]^2$, the steady-state heat equation is

$$-\nabla\cdot\bigl(h(x,y)\,\nabla T\bigr) = q(x,y),$$

with a Dirichlet condition on the bottom wall and insulated (zero-flux Neumann) conditions
on the other three walls:

$$T(x, y{=}0) = T_0, \qquad \nabla T\cdot\hat{n} = 0 \ \text{ on } \ \partial\Omega\setminus\{y=0\}.$$

$T_0$ defaults to $300\,\mathrm{K}$ (`DBC_value`). The zero-flux Neumann condition is
*natural* — it needs no explicit term in the weak form.

Discretizing with P1 Lagrange elements on a quadrilateral or triangular mesh gives the
weak form implemented in {py:class}`forward_solver.SteadyHeat2DForwardSolver`:

$$a(u,v) = \int_\Omega h\,\nabla u\cdot\nabla v \ \mathrm{d}x,
\qquad
L(v) = \int_\Omega q\,v \ \mathrm{d}x.$$

The resulting system is symmetric positive definite (guaranteed by $h > 0$), which is why
the default KSP configuration is conjugate gradient with a `hypre` preconditioner.

## Observation model

A synthetic measurement is produced by adding uncorrelated Gaussian noise to the forward
solution:

$$T_{\rm obs}(x,y) = T(x,y) + \varepsilon, \qquad \varepsilon \sim \mathcal{N}(\mu, \sigma^2).$$

See {py:meth}`forward_solver.SteadyHeat2DForwardSolver.add_noise`. The noise is drawn on
rank 0 and broadcast, so the observation is identical across MPI ranks.

## Inverse problem

Recover $h$ by minimizing the regularized misfit

$$J[T(h), h] = \tfrac{1}{2}\left[\int_\Omega \frac{\bigl(T(h) - T_{\rm obs}\bigr)^2}{\sigma^2}\ \mathrm{d}x
\; + \; \alpha \int_\Omega \lvert\nabla h\rvert^2 \ \mathrm{d}x \right]$$

subject to the PDE constraint and the positivity bound $h \ge h_{\min}$.

The second term is **Tikhonov ($H^1$) regularization**: it penalizes roughness in $h$ and
is what makes the otherwise ill-posed reconstruction stable against sensor noise. The
ratio $\alpha/\sigma^2$ sets the trade-off between fitting the data and smoothing the
solution — too small and noise is imprinted onto $h$, too large and genuine structure is
smoothed away.

## Discrete adjoint gradient

Introduce the Lagrangian with multiplier (adjoint state) $\lambda$,

$$\mathcal{L} = J + \langle\lambda, g[T(h), h]\rangle,
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

## Log parametrization

By default ({py:class}`tao_solver.SteadyHeat2DTAOSolver`, `use_logh=True`) the
optimization variable is

$$m = \log h,$$

which enforces positivity structurally and conditions the problem better when $h$ spans
orders of magnitude. The chain rule is applied to the gradient,

$$\frac{\mathrm{d}J}{\mathrm{d}m} = \frac{\mathrm{d}J}{\mathrm{d}h}\,\frac{\mathrm{d}h}{\mathrm{d}m}
= h\,\frac{\mathrm{d}J}{\mathrm{d}h},$$

and the bounds are mapped to $[\log h_{\min}, \log h_{\max}]$. With `use_logh=False` the
optimization runs directly in $h$ with bounds $[h_{\min}, h_{\max}]$.

## Optimization loop

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
