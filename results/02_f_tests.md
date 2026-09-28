# 02 F-tests (in-sample, one row per athlete)

## Baseline (Driveline 4)
Overall F(4,536) = 108.1, p = 1.7e-67;
R2 = 0.447, adj R2 = 0.442

| Input dropped | Partial F | p | R2 lost | HC3 robust p (max) |
|---|---|---|---|---|
| SJ peak power | F(1,536) = 69.3 | 7.0e-16 | 0.072 | 2.2e-15 |
| CMJ RSI-mod | F(1,536) = 9.8 | 1.9e-03 | 0.010 | 2.5e-03 |
| Hop RSI | F(1,536) = 9.8 | 1.8e-03 | 0.010 | 3.9e-03 |
| IMTP net peak force | F(1,536) = 14.1 | 1.9e-04 | 0.015 | 2.7e-04 |

Checks: Breusch-Pagan p = 0.050 (<0.05 = uneven error spread -> HC3 column);
Shapiro p = 0.000 (residual skew -0.68); max Cook's D = 0.079.

## Final (Driveline 4 + bodyweight curve)
Overall F(6,534) = 82.4, p = 8.7e-73;
R2 = 0.481, adj R2 = 0.475

| Input dropped | Partial F | p | R2 lost | HC3 robust p (max) |
|---|---|---|---|---|
| SJ peak power | F(1,534) = 17.8 | 2.9e-05 | 0.017 | 2.7e-05 |
| CMJ RSI-mod | F(1,534) = 9.9 | 1.7e-03 | 0.010 | 9.3e-04 |
| Hop RSI | F(1,534) = 10.6 | 1.2e-03 | 0.010 | 9.4e-04 |
| IMTP net peak force | F(1,534) = 6.4 | 1.2e-02 | 0.006 | 1.2e-02 |
| Bodyweight curve (BW + BW^2) | F(2,534) = 17.6 | 4.1e-08 | 0.034 | 3.2e-03 |

Checks: Breusch-Pagan p = 0.484 (<0.05 = uneven error spread -> HC3 column);
Shapiro p = 0.000 (residual skew -0.69); max Cook's D = 0.097.
