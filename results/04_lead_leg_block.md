# 04 Lead-leg block (OBP pitching biomechanics, a separate cohort)

- 411 pitches in POI, 411 after joining metadata
- dropped 8 pitches with no force-plate data -> 403 pitches
- 100 pitchers (78 R, 22 L)
- 78 right-handed pitchers; 179 same-level pairs within 5 kg and 5 cm; picked largest velo gap
- sampling: force plate 1080 Hz, joint positions 360 Hz; used as OBP provides them (OBP pre-filtered: 4th-order Butterworth, 40 Hz force / 20 Hz joints); no further filtering here
- H (2916_5): C3D markers vs model ankle 0.5 mm; raw plate sum vs processed peak 0.0%; landing plates [0, 2]
- L (2907_2): C3D markers vs model ankle 0.4 mm; raw plate sum vs processed peak 0.2%; landing plates [0, 2]

## Does peak lead-leg force track velo? (n = 100 pitchers, seed 0)
| Metric (per-pitcher mean) | Pearson r | 95% bootstrap CI | p | Spearman |
|---|---|---|---|---|
| Lead-leg peak GRF (x bodyweight) | 0.356 | 0.19 to 0.51 | 2.8e-04 | 0.307 |
| Lead-leg peak GRF (N) | 0.457 | 0.32 to 0.59 | 1.7e-06 | 0.437 |
| Rear-leg peak GRF (x bodyweight) | 0.158 | -0.02 to 0.33 | 1.2e-01 | 0.149 |
| Body mass (kg) | 0.266 | 0.07 to 0.45 | 7.5e-03 | 0.279 |

Partial r, lead-leg GRF (N) vs velo controlling for body mass: 0.39

## Illustrative pair (selected on level, handedness, mass, height and velo only)
| tag | session_pitch | level | velo | mass_kg | height_m | peak_bw | angle_deg | poi_angle_deg | peak_ms_vs_BR |
|---|---|---|---|---|---|---|---|---|---|
| H | 2916_5 | college | 94.00 | 86.64 | 1.85 | 2.38 | 55.31 | 55.31 | -43.60 |
| L | 2907_2 | college | 79.10 | 82.56 | 1.85 | 2.18 | 63.60 | 63.60 | -25.90 |

`angle_deg` = force angle above horizontal in the side (x-z) view, recomputed here; `poi_angle_deg` = OBP's value.
`peak_ms_vs_BR` = time of peak lead-leg force relative to ball release (negative = before).
