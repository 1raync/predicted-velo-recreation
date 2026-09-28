# 03 Bodyweight variants (held-out, 10x5-fold CV, n = 541)

| Model | R2 | R2 p5 | R2 p95 | MAE (mph) | Beats baseline | Max VIF |
|---|---|---|---|---|---|---|
| Driveline 4 (baseline) | 0.425 | 0.321 | 0.524 | 3.977 | — | 2.233 |
| + BW (straight line) | 0.424 | 0.314 | 0.528 | 3.957 | 64% | 4.784 |
| + BW curve (BW + BW^2)  [final] | 0.457 | 0.341 | 0.576 | 3.824 | 96% | 4.809 |
| SJ -> SJ/kg | 0.362 | 0.234 | 0.464 | 4.188 | 4% | 2.756 |
| IMTP -> IMTP/kg | 0.415 | 0.310 | 0.514 | 4.013 | 14% | 2.328 |
| All per-kg (SJ/kg, IMTP/kg) | 0.286 | 0.179 | 0.389 | 4.441 | 0% | 2.544 |
| Bodyweight only | 0.147 | 0.003 | 0.287 | 4.748 | 0% |  |

**Curve vs baseline:** F(2,534) = 17.6, p = 4.1e-08.
HC3 robust p: BW = 3.2e-03, BW^2 = 7.9e-09.
Curve peaks at ~212 lb (95th percentile bodyweight = 237 lb).
