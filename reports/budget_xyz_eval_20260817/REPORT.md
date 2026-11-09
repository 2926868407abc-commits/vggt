# MIRAGE-3D budgeted attack: sitting_xyz evaluation

Date: 2026-08-17

## Protocol

- Sequence: `rgbd_dataset_freiburg3_sitting_xyz`, 10 frames.
- Victim: frozen VGGT; the optimizer updates the patch texture only.
- EOT: disabled for this comparison.
- Pose metric: TUM ATE/RPE with `align=True, correct_scale=True`.
- Normalized attack strength: ATE divided by the random-trajectory ceiling (`0.200452 m`).
- Output checks: matched-sequence clean thresholds for confidence spread, low-confidence fraction, point/depth-head disagreement, and pose-depth reprojection.
- Budget constraints additionally include a differentiable tracking error in pixels.
- Point-map metrics use Umeyama Sim(3) followed by ICP. Depth uses the evaluator's scale-only alignment; the table reports the `median-scale` configuration.

## Main results

Lower is better for ATE, RPE, Acc, Comp, and AbsRel. Percentages in parentheses are changes relative to clean.

| run | objective | budget fraction | ATE (m) | ATE/ceiling | RPE trans | RPE rot (deg) | Acc | Comp | AbsRel | output filters | train/deploy gap |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| clean | — | — | 0.007910 | 3.95% | 0.011934 | 0.383853 | 0.038548 | 0.067175 | 0.050580 | — | — |
| `budget_xyz_u_f05_o2_250` | direct untargeted ATE | 0.50 | 0.019329 (+144.38%) | **9.64%** | 0.025055 (+109.94%) | 0.409644 (+6.72%) | 0.046538 (+20.73%) | 0.062955 (-6.28%) | 0.053868 (+6.50%) | **0/4** | +0.7% |
| `budget_xyz_u_f08_o2_250` | direct untargeted ATE | 0.80 | 0.015777 (+99.47%) | 7.87% | 0.021275 (+78.26%) | 0.393430 (+2.50%) | 0.043102 (+11.82%) | 0.063185 (-5.94%) | 0.051943 (+2.69%) | **0/4** | +0.8% |
| `budget_xyz_t_f08_o1_250` | targeted piecewise gauge | 0.80 | 0.009729 (+23.00%) | 4.85% | 0.012844 (+7.62%) | 0.378919 (-1.29%) | 0.041480 (+7.61%) | 0.064129 (-4.53%) | 0.052083 (+2.97%) | **0/4** | -0.0% |
| `budget_xyz_t_f08_o2_250` | targeted piecewise gauge | 0.80 | 0.013768 (+74.07%) | 6.87% | 0.013491 (+13.04%) | 0.381166 (-0.70%) | 0.043491 (+12.82%) | 0.067567 (+0.58%) | 0.052351 (+3.50%) | **0/4** | -0.0% |

## Alignment and constraint checks

| run | Sim(3)-aligned ATE | SE(3)-aligned ATE | unaligned ATE | gauge absorbed | final tracking proxy / safe bound |
|---|---:|---:|---:|---:|---:|
| clean | 0.007910 | 0.152252 | 3.387260 | — | 0 / — |
| `budget_xyz_u_f05_o2_250` | 0.019329 | 0.146756 | 3.384279 | 65.7% | 0.458 / 1.000 px |
| `budget_xyz_u_f08_o2_250` | 0.015777 | 0.148763 | 3.385453 | 61.4% | 0.336 / 1.600 px |
| `budget_xyz_t_f08_o1_250` | 0.009729 | 0.142387 | 3.381110 | 94.8% | 0.449 / 1.600 px |
| `budget_xyz_t_f08_o2_250` | 0.013768 | 0.154780 | 3.388746 | 60.9% | 0.430 / 1.600 px |

The raw unaligned ATE is dominated by VGGT's arbitrary global frame and is not an attack-quality metric. The useful checks are the production Sim(3)-aligned ATE and `gauge_absorbed`: the strongest run still leaves 34.3% of its clean-relative trajectory change after fitting one global Sim(3), so its ATE increase is not a pure global-gauge artifact.

## Conclusions

1. The budgeted formulation escapes the strict-clean solution: the best strict four-head run was about 4.2% of ceiling, while the best budgeted run reaches 9.64% and passes all four deployment filters.
2. Directly maximizing the aligned pose error is better than targeting the current hand-designed piecewise gauge on `sitting_xyz`.
3. The 50% budget run is stronger than the looser 80% run. This is an optimization/trajectory effect, not evidence that tighter constraints are always better; it should be repeated across seeds.
4. The best attack mainly damages translation: ATE rises 144% and translational RPE rises 110%, while rotational RPE rises only 6.7%.
5. It also worsens point-map Acc and depth AbsRel, but point-map Comp improves. Therefore this is a multi-head perturbation, not yet a result where every reconstruction metric degrades simultaneously.
6. All train/deploy gaps are below 1%, so the result is not caused by a mismatch between the training forward pass and the saved evaluated prediction.

## Next experiment justified by this evaluation

- Carry `untargeted_gt`, budget fraction `0.50`, order `2` to `sitting_halfsphere` first.
- Run at least two seeds for the `0.50` and `0.80` budgets on `sitting_xyz` before treating their ordering as stable.
- Keep EOT off for that comparison; add photometric EOT only after the no-EOT Pareto point is reproduced. Geometric EOT remains excluded because the earlier gradient-SNR diagnostic showed a large direction change, especially on `sitting_halfsphere`.
