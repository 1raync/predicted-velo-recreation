"""Step 1: recreate Driveline's 4-input multiple linear regression.
Outputs results/01_baseline.md"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from common import load, folds, BASE, LABELS, TARGET, RES, md_table


def cv_scores(df, feats, fs):
    r2, mae = [], []
    for tr, te in fs:
        m = LinearRegression().fit(df.loc[tr, feats], df.loc[tr, TARGET])
        p = m.predict(df.loc[te, feats])
        r2.append(r2_score(df.loc[te, TARGET], p))
        mae.append(mean_absolute_error(df.loc[te, TARGET], p))
    return np.array(r2), np.array(mae)


def vif(X):
    out = {}
    for c in X.columns:
        o = X.drop(columns=c)
        out[c] = 1 / (1 - LinearRegression().fit(o, X[c]).score(o, X[c]))
    return pd.Series(out)


if __name__ == "__main__":
    df, n_raw = load()
    fs = folds(df)
    r2, mae = cv_scores(df, BASE, fs)
    single = {LABELS[f]: cv_scores(df, [f], fs)[0].mean() for f in BASE}
    Z = (df[BASE] - df[BASE].mean()) / df[BASE].std()
    std = LinearRegression().fit(Z, df[TARGET]).coef_
    v = vif(df[BASE])

    t = pd.DataFrame({"Input": [LABELS[f] for f in BASE], "R2 alone (held-out)": [single[LABELS[f]] for f in BASE],
                      "mph per 1 SD (in 4-input model)": std, "VIF": v.values})
    lv = df["playing_level"].value_counts()
    txt = f"""# 01 Baseline: Driveline's 4 inputs

**Cohort:** {len(df)} athletes (one row each, first test) from {n_raw} raw test rows.
Levels: {', '.join(f'{k} {v}' for k, v in lv.items())}. Test dates {df.test_date.min()} to {df.test_date.max()}.

**Held-out performance (10x5-fold CV):**
- R2 = {r2.mean():.3f} (5-95%: {np.percentile(r2,5):.3f}-{np.percentile(r2,95):.3f})
- MAE = {mae.mean():.2f} mph
- Driveline's published model (2021): R2 0.54, MAE 2.7 mph

{md_table(t)}
"""
    (RES / "01_baseline.md").write_text(txt, encoding="utf-8")
    print(txt)
