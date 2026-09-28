# 01 Baseline: Driveline's 4 inputs

**Cohort:** 541 athletes (one row each, first test) from 1934 raw test rows.
Levels: College 267, High School 223, Pro 51. Test dates 2023-01-31 to 2024-08-13.

**Held-out performance (10x5-fold CV):**
- R2 = 0.425 (5-95%: 0.321-0.524)
- MAE = 3.98 mph
- Driveline's published model (2021): R2 0.54, MAE 2.7 mph

| Input | R2 alone (held-out) | mph per 1 SD (in 4-input model) | VIF |
|---|---|---|---|
| SJ peak power | 0.375 | 2.670 | 2.233 |
| CMJ RSI-mod | 0.277 | 0.969 | 2.093 |
| Hop RSI | 0.080 | 0.756 | 1.263 |
| IMTP net peak force | 0.255 | 1.066 | 1.748 |
