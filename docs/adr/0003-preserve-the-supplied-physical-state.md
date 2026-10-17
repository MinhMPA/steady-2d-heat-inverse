# Preserve the supplied physical state during new inference calls

The new inference workflow returns its optimized field or posterior draws while
restoring the supplied forward model to its pre-call physical state on return
or handled interruption. Applying a selected draw or summary is explicit and
recomputes its temperature, because sampling has no single natural final field
and exposing a trial conductivity with an unrelated temperature would mislead
later plots and calculations. The user accepted this behavior in Q9; original
TAO calls keep their established behavior under Q2.
