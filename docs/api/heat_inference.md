# Conductivity posterior interfaces

The scientific sequence is a fixed heat experiment and sensor likelihood, a
proper full-field prior, a complete inferred state, and saved physical fields.
See {doc}`../hmc` for interpretation and {doc}`../installation` for optional
sampling and diagnostic dependencies. Imports remain flat top-level modules.

## Declared experiment and complete state

```{eval-rst}
.. automodule:: heat_inference
   :members: HeatInverseProblem, InferredState, PosteriorEstimate
```

## Physical observations and prior

```{eval-rst}
.. automodule:: temperature_observations
   :members: TemperatureObservations, SensorOperator

.. automodule:: conductivity_prior
   :members: GaussianLogConductivity, BoundGaussianPrior

.. automodule:: heat_model
   :members: HeatNumerics, NodalField
```

## Saved physical fields and convergence evidence

```{eval-rst}
.. automodule:: posterior_samples
   :members: PosteriorSamples, FieldSummary

.. automodule:: posterior_diagnostics
   :members: DiagnosticReport
```

## Trajectory controls

```{eval-rst}
.. automodule:: hmc_blackjax
   :members: HMCControls, require_sampling_environment
```
