"""Step 3: ways to use bodyweight. Outputs results/03_bodyweight.md"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from common import load, folds, BASE, FEATS_J, TARGET, RES, md_table
from s01_baseline_model import cv_scores, vif

VARIANTS = {
    "Driveline 4 (baseline)": BASE,
    "+ BW (straight line)": BASE + ["BW_c"],
    "+ BW curve (BW + BW^2)  [final]": FEATS_J,
    "SJ -> SJ/kg": ["SJ_PP_perkg", "CMJ_RSImod", "Hop_RSI", "IMTP_NPF"],
    "IMTP -> IMTP/kg": ["SJ_PP", "CMJ_RSImod", "Hop_RSI", "IMTP_perkg"],
    "All per-kg (SJ/kg, IMTP/kg)": ["SJ_PP_perkg", "CMJ_RSImod", "Hop_RSI", "IMTP_perkg"],
    "Bodyweight only": ["BW_c"],
}

if __name__ == "__main__":
    df, _ = load()
    fs = folds(df)
    base_r2, _ = cv_scores(df, BASE, fs)
    rows = []
    for k, f in VARIANTS.items():
        r2, mae = cv_scores(df, f, fs)
        rows.append([k, r2.mean(), np.percentile(r2, 5), np.percentile(r2, 95), mae.mean(),
                     ("—" if k.startswith("Driveline 4") else f"{np.mean(r2 > base_r2):.0%}"), vif(df[f]).max() if len(f) > 1 else np.nan])
    t = pd.DataFrame(rows, columns=["Model", "R2", "R2 p5", "R2 p95", "MAE (mph)", "Beats baseline", "Max VIF"])
    y = df[TARGET]
    A = sm.OLS(y, sm.add_constant(df[BASE])).fit()
    J = sm.OLS(y, sm.add_constant(df[FEATS_J])).fit()
    F, p, dfd = J.compare_f_test(A)
    Jh = sm.OLS(y, sm.add_constant(df[FEATS_J])).fit(cov_type="HC3")
    b1, b2 = J.params["BW_c"], J.params["BW_c2"]
    peak_lb = (df["BW_kg"].mean() - b1 / (2 * b2)) * 2.2046
    txt = f"""# 03 Bodyweight variants (held-out, 10x5-fold CV, n = {len(df)})

{md_table(t)}

**Curve vs baseline:** F({int(dfd)},{int(J.df_resid)}) = {F:.1f}, p = {p:.1e}.
HC3 robust p: BW = {Jh.pvalues['BW_c']:.1e}, BW^2 = {Jh.pvalues['BW_c2']:.1e}.
Curve peaks at ~{peak_lb:.0f} lb (95th percentile bodyweight = {df['BW_lb'].quantile(.95):.0f} lb).
"""
    (RES / "03_bodyweight.md").write_text(txt, encoding="utf-8")
    print(txt)
