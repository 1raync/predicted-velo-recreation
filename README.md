# Driveline's OBP: Predicting Pitch Velocity from Force-Plate Tests: an Independent Recreation

An independent recreation of the **Predicted Pitch Velocity** model described by
[Driveline Baseball (2021)](https://drivelinebaseball.com/blogs/blog/predicted-pitch-velocity),
rebuilt from scratch on the public
[OpenBiomechanics Project](https://github.com/drivelineresearch/openbiomechanics) (OBP)
High Performance dataset, plus one extension: **how bodyweight should (and shouldn't) enter the model.**

> Personal, non-commercial project. Not affiliated with or endorsed by Driveline Baseball.

![Two athletes with the same predicted velo but opposite gaps](figures/two_athlete_gap.png)

## TL;DR

| | Driveline (2021, published) | My recreation (Driveline's 4 inputs) | + bodyweight curve |
|---|---|---|---|
| Held-out R² | 0.54 | **0.43** (5–95%: 0.32–0.52) | **0.46** (0.34–0.58) |
| Mean absolute error | 2.7 mph | 3.98 mph | 3.82 mph |
| Athletes | not reported | 541 | 541 |

- **Squat-jump peak power is the dominant input**, consistent with Driveline's statement that it is the metric most correlated with pitch velocity.
- **Relative (per-kg) strength makes predictions worse.** Held-out R² falls to 0.29 when both power and strength are divided by bodyweight.
- **Adding bodyweight as a curve helps.** It beats the baseline in 96% of cross-validation fits (partial F(2,534) = 17.6, p = 4×10⁻⁸). At equal strength, heavier athletes throw harder up to about 212 lb, then the effect levels off.

## Background

I loved Driveline's idea: that fitness tests can model the pitching velo an athlete's body "should" produce. The **gap** between predicted and actual velo then points training. Athletes throwing *above* prediction shift toward the weight room; athletes throwing *below* it shift toward throwing and mechanics work. Their published model was a multiple linear regression with a 75/25 train/test split, VIF < 5, and residual checks. The four inputs "weighted the heaviest" were:

| Test | Input | Plain meaning |
|---|---|---|
| Squat Jump (SJ) | Peak power (W) | Push power from a paused squat, no bounce |
| Countermovement Jump (CMJ) | RSI-modified | Jump height ÷ time to take off |
| 10/5 Hop test | RSI (flight ÷ contact time) | Reactive "springiness" |
| Isometric Mid-Thigh Pull (IMTP) | Net peak force (N) | Maximal pulling strength |

## Data

- **Source:** OBP `high_performance/hp_obp.csv`, pinned to commit
  [`44b98da`](https://github.com/drivelineresearch/openbiomechanics/tree/44b98dae05cceb016f080ab39d105c85b8639084/high_performance)
  and downloaded by `src/fetch_data.py`. The data is not stored in this repo.
- **Cohort:** 541 athletes (223 high school, 267 college, 51 pro), tested Jan 2023 – Aug 2024,
  from 1,934 raw test rows.
- **Cleaning** (all in `src/common.py`):
  - Kept rows with all four inputs, pitch velo, and bodyweight.
  - Dropped velo < 60 mph (includes 0-mph placeholders).
  - Dropped one physically implausible RSI-mod value.
  - Kept each athlete's **first test only**, so no athlete appears twice.
- **Column note:** OBP includes two hop-test RSI columns. This project uses the flight-time/contact-time ratio,
  which is Driveline's definition. The `jump_height/contact_time` column is a different metric with a different scale.

## Method

- **Model:** ordinary least squares multiple linear regression (scikit-learn / statsmodels).
- **Validation:** 10× repeated 5-fold cross-validation (50 fits). Every held-out number in this README is the
  mean across those fits. Since each athlete has one row, no athlete is ever in both the training and test folds.
- **Feature impact:** permutation importance, i.e. the drop in held-out R² when one input is shuffled. The two
  bodyweight terms are shuffled together.
- **Inference:** overall and partial F-tests, checked with HC3 heteroscedasticity-robust errors, Breusch-Pagan,
  Shapiro-Wilk and Cook's distance.
- **Bodyweight curve uncertainty:** 1,000 bootstrap refits resampling athletes.

## Results

### Feature impact
![Feature impact](figures/feature_impact.png)

In the final model, SJ peak power ranks #1 in 82% of fits. Bodyweight is a clear #2. The three remaining
Driveline inputs fall in the same order as Driveline's list, but their ranges overlap, so that ordering is not firm.
In the 4-input model without bodyweight, IMTP carries more weight. That suggests part of its signal was standing in
for body size.

### Bodyweight
![Bodyweight curve](figures/bodyweight_curve.png)

| Model | Held-out R² | MAE (mph) |
|---|---|---|
| All per-kg (SJ/kg, IMTP/kg) | 0.29 | 4.44 |
| SJ → SJ/kg | 0.36 | 4.19 |
| Driveline 4 (baseline) | 0.43 | 3.98 |
| + bodyweight (straight line) | 0.42 | 3.96 |
| **+ bodyweight curve (final)** | **0.46** | **3.82** |

Dividing by bodyweight penalizes heavier athletes, who, at the same power output, tend to throw harder. A
straight-line bodyweight term adds nothing. A curve does. The dip above ~212 lb is **not** statistically supported:
past the peak the bootstrap band is wide enough to fit a flat curve, and only 5% of athletes weigh more than 237 lb.

### F-tests
All four Driveline inputs remain significant in the final model (all p < 0.05, including with robust errors).
Full tables: [`results/01_baseline.md`](results/01_baseline.md),
[`results/02_f_tests.md`](results/02_f_tests.md), [`results/03_bodyweight.md`](results/03_bodyweight.md).

### Why the recreation scores lower than Driveline's published model
Likely reasons, none of which can be confirmed from public information:
- **Different target:** Driveline used motion-capture velo, while OBP's `pitch_speed_mph` is undocumented.
- **Different cohort:** 2023–24 athletes vs. Driveline's pre-2021 sample.
- **Unknown extra inputs:** Driveline listed only the inputs "weighted heaviest."
- **Evaluation method:** Driveline reported a single 75/25 split. Across 50 fits here, R² ranges from 0.32 to 0.52.

## Limitations
- No height, age, limb length or handedness in the data. Bodyweight likely stands in partly for body size.
- Associations only. Nothing here shows that raising a test score *causes* higher velo.
- Residuals are left-skewed: some athletes throw far below prediction, which is exactly the "skill gap" case.
- OBP column definitions are intentionally undocumented, and the OBP README asks users not to infer formulas.
- One dataset, with no external validation.

## Reproduce

```bash
git clone https://github.com/1raync/predicted-velo-recreation.git
cd predicted-velo-recreation
pip install -r requirements.txt
python run_all.py        # downloads data, writes results/ and figures/ (~1-2 min)
```

## Attribution & licenses
- **Data:** The OpenBiomechanics Project, Driveline Baseball Research & Development,
  <https://github.com/drivelineresearch/openbiomechanics>, licensed
  [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). OBP's license also includes a
  **professional-organization exclusion**. Read
  [LICENSE-DATA.md](https://github.com/drivelineresearch/openbiomechanics/blob/main/LICENSE-DATA.md)
  before using the data. Please cite OBP via its
  [CITATION.cff](https://github.com/drivelineresearch/openbiomechanics/blob/main/CITATION.cff).
- **Figures and results** in `figures/` and `results/` are derived from OBP data and are shared under
  CC BY-NC-SA 4.0 (see [`LICENSE-DATA.md`](LICENSE-DATA.md)).
- **Code** in this repo: MIT (see [`LICENSE`](LICENSE)).
- Method inspired by Driveline Baseball's public blog posts. This project is not affiliated with or endorsed by Driveline.

## References
- Driveline Baseball. *Predicted Pitch Velocity* (2021). <https://drivelinebaseball.com/blogs/blog/predicted-pitch-velocity>
- Driveline Baseball. *High Performance Assessment: Strength Testing Using Force Plates* (2020). <https://www.drivelinebaseball.com/2020/11/high-performance-assessment-strength-testing-using-force-plates/>
- Driveline Baseball. *Free Strength Training Program* (2021). <https://drivelinebaseball.com/blogs/blog/free-strength-training-program-baseball>
- The OpenBiomechanics Project. <https://github.com/drivelineresearch/openbiomechanics>
