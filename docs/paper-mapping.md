# Mapping the paper to code

| Paper object | Code |
|---|---|
| Self-centered fourth-order bound | `certificates.self_centered_tv_bound` |
| Residual certificate | `certificates.residual_tv_bound` |
| Distribution-free empirical certificate | `certificates.monte_carlo_tv_bound` |
| One-shot diameter-scaled update | `certificates.one_shot_diameter_tv_bound` |
| Product path-space budget | `certificates.path_budget` |
| Constant-ratio finite-horizon rule | `certificates.constant_ratio_for_path_budget` |
| Amortized `varrho_n`, `H_k`, `M_k`, `delta_k` schedule | `schedules.AmortizedSchedule` |
| Exact exterior Poisson return | `boundaries.poisson_return_delta` and Numba kernel |
| Frozen approximate-law probes | backend `probe` / compiled `probe_batch` |
| Sample splitting | `calibration.validate_center` and backend orchestration |
| Diameter-scaled controlled growth | Numba `DIAMETER_UPPER_SCALE` path |

## Important implementation distinction

The theorem uses the exact target diameter `D(K)`. Computing it after every attachment would
be unnecessarily expensive. The production backend uses the diagonal of the target's bounding
box as an upper bound. Thus the actual death radius is at least the radius prescribed by the
configured `varrho`; the theoretical certificate remains conservative.

The repository does not infer a full infinite-history budget from one run unless every paper
assumption and prefix treatment has been supplied. It records local block data so that the
analysis scripts can construct the relevant budget explicitly.
