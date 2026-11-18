# Share the conductivity posterior between optimization and sampling

The new inference workflow will define the physical heat model, temperature
observations, noise model, and prior once, and use that same posterior for both
optimization and HMC. Existing reconstruction calls retain their original
meaning and numerical behavior through an explicitly separate route, because
silently replacing their integrated misfit and roughness penalty would change
previous scientific results. This trades a single universal calling convention
for an honest distinction between reproducing the original reconstruction and
studying a declared posterior; the user accepted this choice in Q1–Q2 on
2026-10-17.
