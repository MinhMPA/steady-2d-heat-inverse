# Keep the conductivity prior separate from the heat and observation models

Use the Gaussian prior as the first sampling model while retaining the ability
to study sharp conductivity variations with other priors later. Changing that
prior must not require redefining the heat equation or temperature measurements;
however, a future prior may require additional unknown fields, different
coordinates, or a different sampling method. The user accepted this separation
in Q6 on 2026-10-17, explicitly allowing imperfect truth recovery, particularly
with Gaussian priors: inferred smoothing and uncertainty are scientific results
to assess, not grounds for forcing the reconstruction to match the truth.
