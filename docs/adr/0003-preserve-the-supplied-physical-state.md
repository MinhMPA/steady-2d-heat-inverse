# Preserve the supplied physical state during new inference calls

The new inference workflow returns its optimized field or posterior draws while
restoring the supplied forward model to its pre-call physical state on return
or handled interruption. Applying a selected draw or summary is explicit and
recomputes its temperature, because sampling has no single natural final field
and exposing a trial conductivity with an unrelated temperature would mislead
later plots and calculations. The user accepted this behavior in Q9; original
TAO calls keep their established behavior under Q2.

Q24, accepted on 2026-10-26, clarifies that ordinary application first checks
the destination mesh, heat source, and boundary conditions against the recorded
experiment. A mismatch is reported before changing the destination model.
Its current conductivity and temperature are the state to be replaced and are
excluded from this comparison.
Conductivity arrays remain available for explicitly declared new experiments;
their changed heating conditions must not be mistaken for those used in the
original inference. Automatic transfer between meshes is outside the first
release.
