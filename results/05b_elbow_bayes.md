# 05b Elbow stress beyond expected: hierarchical Bayesian check of the s05 split (NumPyro)

Model (all pitches, not pitcher means): varus = expected(velo, mass, height) + u_pitcher + noise, u_pitcher ~ Normal(0, tau).
Score = posterior-mean u_pitcher; thirds by rank, as in s05. NUTS, 4 chains x 2000 draws, seed 0.

- 411 pitches -> 407 after s05's event-timing drop; 100 pitchers with 2-5 pitches each
- REML random-intercept model vs Gaussian fit: slopes 1.29, 1.09, -2.63 vs 1.28, 1.08, -2.01 Nm per mph, kg, m; tau 13.52 vs 13.64 Nm; pitcher effects r = 1.0000, max |diff| 0.18 Nm
- Gaussian: 6 of the 6 s05 metrics with q < 0.05 still have q < 0.05; newly significant: ['shoulder_abduction_fp']
- Student-t: 6 of the 6 s05 metrics with q < 0.05 still have q < 0.05; newly significant: ['glove_shoulder_abduction_mer', 'pelvis_lateral_tilt_fp']

## Fits (posterior mean [95% interval])
| noise | max R-hat | min ESS | divergences | Nm per mph | Nm per kg | Nm per m | tau (Nm) | sigma (Nm) | nu | r with OLS resid |
|---|---|---|---|---|---|---|---|---|---|---|
| Gaussian | 1.0002 | 8028 | 0 | 1.28 [0.70, 1.88] | 1.08 [0.78, 1.38] | -2.01 [-43.85, 39.88] | 13.64 [11.82, 15.82] | 4.45 [4.11, 4.82] |  | 1.0000 |
| Student-t | 1.0012 | 5384 | 0 | 1.23 [0.66, 1.79] | 1.09 [0.80, 1.38] | -0.79 [-40.38, 40.00] | 13.01 [11.26, 15.03] | 1.73 [1.46, 2.03] | 2.06 [1.56, 2.68] | 0.9887 |

## Group agreement with s05
| noise | same group as s05 (of 100) | switched (s05 -> Bayes) | HIGH with P(top third) >= 0.8 | LOW with P(bottom third) >= 0.8 |
|---|---|---|---|---|
| Gaussian | 96 | 922 (LOW->MID, 2 pitches), 956 (MID->HIGH, 3 pitches), 986 (HIGH->MID, 3 pitches), 1562 (MID->LOW, 4 pitches) | 28 of 34 | 26 of 34 |
| Student-t | 92 | 922 (LOW->MID, 2 pitches), 956 (MID->HIGH, 3 pitches), 964 (LOW->MID, 5 pitches), 970 (LOW->MID, 3 pitches), 1562 (MID->LOW, 4 pitches), 1592 (MID->LOW, 5 pitches), 1633 (HIGH->MID, 5 pitches), 1691 (MID->LOW, 5 pitches) | 30 of 34 | 26 of 34 |

## The s05 metrics with FDR q < 0.05, under each grouping (d = Cohen's d; q from 5000 permutations + BH FDR)
| metric | OLS (s05) d | OLS (s05) q | Gaussian d | Gaussian q | Student-t d | Student-t q |
|---|---|---|---|---|---|---|
| shoulder_internal_rotation_moment | 1.723 | 0.005 | 1.758 | 0.005 | 1.689 | 0.005 |
| max_shoulder_external_rotation | -1.114 | 0.005 | -1.403 | 0.005 | -1.291 | 0.005 |
| shoulder_absorption_fp_br | 1.111 | 0.005 | 1.169 | 0.005 | 1.126 | 0.005 |
| max_elbow_extension_velo | 0.958 | 0.006 | 0.941 | 0.010 | 0.928 | 0.007 |
| stride_angle | -0.920 | 0.006 | -0.897 | 0.010 | -0.771 | 0.037 |
| torso_rotation_br | 0.887 | 0.012 | 0.876 | 0.010 | 0.803 | 0.018 |

Synthetic recovery check (simulate from each fitted model, refit, compare with the known truth): run
`python src/s05b_elbow_bayes.py --synthetic` (~20+ min), which writes results/05b_synthetic.md (kept separate so
run_all doesn't overwrite it).
