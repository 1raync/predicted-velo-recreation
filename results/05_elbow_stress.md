# 05 Elbow stress beyond expected (OBP pitching biomechanics, a separate cohort)

- 411 pitches; dropped 4 with broken event timing ['2918_2', '2918_3', '2918_4', '2923_3'] -> 407
- 100 pitchers; expected varus = -91.0 + 1.29*mph + 1.09*kg -2.7*m (R2 0.51, residual SD 13.5 Nm)
- HIGH = residual >= +4.5 Nm (n=34), LOW = residual <= -5.6 Nm (n=34); middle third dropped
- 75 metrics compared; 6 with FDR q < 0.05
- layback curves: -300..+90 ms around release, linear interpolation onto a common 360 Hz grid aligned to release, per-pitcher means (HIGH 34, LOW 34)
- 13 matched right-handed HIGH/LOW pairs; HIGH has less layback in 13 of 13
- H (pitcher 789, pitch 1313_1): throwing-elbow markers vs model elbow 0.1 mm
- L (pitcher 1606, pitch 2833_3): throwing-elbow markers vs model elbow 0.3 mm

## Group balance (HIGH vs LOW, per-pitcher means)
| metric | HIGH | LOW | d | p_perm |
|---|---|---|---|---|
| pitch_speed_mph | 84.68 | 84.23 | 0.10 | 0.68 |
| elbow_varus_moment | 128.09 | 98.41 | 1.91 | 0.00 |
| session_mass_kg | 92.23 | 91.07 | 0.11 | 0.66 |
| session_height_m | 1.86 | 1.86 | 0.03 | 0.93 |
| age_yrs | 21.33 | 21.30 | 0.01 | 0.95 |

## Selected metrics (d = Cohen's d; p from 5000 permutations, seed 0; q = FDR across all 75 metrics)
| metric | HIGH | LOW | d | p_perm | rho_all | q |
|---|---|---|---|---|---|---|
| Shoulder internal rotation moment (Nm) | 121.103 | 94.562 | 1.723 | 0.000 | 0.598 | 0.005 |
| Max layback (deg) | 164.050 | 173.832 | -1.114 | 0.000 | -0.487 | 0.005 |
| Shoulder energy absorbed, foot plant to release (J) | 31.032 | 16.392 | 1.111 | 0.000 | 0.433 | 0.005 |
| Max elbow extension velo (deg/s) | 2573.902 | 2371.946 | 0.958 | 0.000 | 0.439 | 0.006 |
| Stride angle (deg, + = cross-body) | -0.146 | 4.729 | -0.920 | 0.000 | -0.337 | 0.006 |
| Torso rotation at release (deg) | 123.343 | 115.988 | 0.887 | 0.001 | 0.346 | 0.012 |
| Pelvis-to-torso peak timing (s) | 0.011 | 0.008 | 0.269 | 0.273 | 0.172 | 0.584 |
| Hip-shoulder separation (deg) | 32.864 | 31.364 | 0.235 | 0.335 | 0.096 | 0.644 |

Layback vs peak varus moment, adjusted for velo, mass and height: partial r = -0.46 (p = 1.7e-06).

## Whole layback curve, HIGH vs LOW (spm1d two-sample t-test, two-tailed, alpha 0.05)
- unequal variance (primary): critical t = 3.03; significant clusters: -144 to 32 ms (HIGH lower, p = 8.4e-15)
- equal variance: critical t = 3.03; significant clusters: -144 to 32 ms (HIGH lower, p = 8.7e-15)
- group-mean curves peak at -33 ms: HIGH 163.4 deg vs LOW 173.3 deg

## All metrics with q < 0.15
| metric | HIGH | LOW | d | p_perm | rho_all | q |
|---|---|---|---|---|---|---|
| shoulder_internal_rotation_moment | 121.103 | 94.562 | 1.723 | 0.000 | 0.598 | 0.005 |
| max_shoulder_external_rotation | 164.050 | 173.832 | -1.114 | 0.000 | -0.487 | 0.005 |
| shoulder_absorption_fp_br | 31.032 | 16.392 | 1.111 | 0.000 | 0.433 | 0.005 |
| max_elbow_extension_velo | 2573.902 | 2371.946 | 0.958 | 0.000 | 0.439 | 0.006 |
| stride_angle | -0.146 | 4.729 | -0.920 | 0.000 | -0.337 | 0.006 |
| torso_rotation_br | 123.343 | 115.988 | 0.887 | 0.001 | 0.346 | 0.012 |
| pelvis_lateral_tilt_fp | 1.664 | -1.227 | 0.679 | 0.006 | 0.267 | 0.064 |
| torso_rotation_mer | 108.184 | 101.839 | 0.633 | 0.009 | 0.268 | 0.081 |
| rear_grf_y_max | 139.535 | 104.590 | 0.599 | 0.017 | 0.227 | 0.114 |
| shoulder_abduction_fp | 89.663 | 84.280 | 0.596 | 0.017 | 0.258 | 0.114 |
| glove_shoulder_abduction_mer | 36.831 | 32.090 | 0.589 | 0.018 | 0.302 | 0.114 |
| pelvis_rotation_fp | 38.399 | 31.584 | 0.580 | 0.018 | 0.171 | 0.114 |

## Drawn pair (largest residual gap among the matched pairs; pitcher means, plus the drawn pitch)
| tag | user | session_pitch | level | velo | mass_kg | height_m | varus_nm | expected_nm | resid_nm | layback_deg | pitch_varus_nm | pitch_layback_deg | frame_ms_vs_BR |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| H | 789 | 1313_1 | college | 85.8 | 84.4 | 1.8 | 129.5 | 106.4 | 23.0 | 160.6 | 122.3 | 158.3 | -30.6 |
| L | 1606 | 2833_3 | college | 85.6 | 83.9 | 1.8 | 85.8 | 105.6 | -19.9 | 181.0 | 83.5 | 178.8 | -30.6 |
