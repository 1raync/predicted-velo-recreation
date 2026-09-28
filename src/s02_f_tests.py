"""Step 2: F-tests and assumption checks. Outputs results/02_f_tests.md"""
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
from common import load, BASE, FEATS_J, LABELS, TARGET, RES, md_table

if __name__ == "__main__":
    df, _ = load()
    y = df[TARGET]
    out = ["# 02 F-tests (in-sample, one row per athlete)\n"]
    for name, feats in [("Baseline (Driveline 4)", BASE), ("Final (Driveline 4 + bodyweight curve)", FEATS_J)]:
        X = sm.add_constant(df[feats])
        full = sm.OLS(y, X).fit()
        hc3 = sm.OLS(y, X).fit(cov_type="HC3")
        rows = []
        drop_sets = {LABELS[f]: [f] for f in BASE}
        if "BW_c" in feats:
            drop_sets["Bodyweight curve (BW + BW^2)"] = ["BW_c", "BW_c2"]
        for lab, cols in drop_sets.items():
            red = sm.OLS(y, sm.add_constant(df[[c for c in feats if c not in cols]])).fit()
            F, p, dfd = full.compare_f_test(red)
            rows.append([lab, f"F({int(dfd)},{int(full.df_resid)}) = {F:.1f}", f"{p:.1e}",
                         full.rsquared - red.rsquared, f"{hc3.pvalues[cols].max():.1e}"])
        bp = het_breuschpagan(full.resid, X)[1]
        sw = stats.shapiro(full.resid)[1]
        cook = full.get_influence().cooks_distance[0].max()
        t = pd.DataFrame(rows, columns=["Input dropped", "Partial F", "p", "R2 lost", "HC3 robust p (max)"])
        out.append(f"""## {name}
Overall F({int(full.df_model)},{int(full.df_resid)}) = {full.fvalue:.1f}, p = {full.f_pvalue:.1e};
R2 = {full.rsquared:.3f}, adj R2 = {full.rsquared_adj:.3f}

{md_table(t)}

Checks: Breusch-Pagan p = {bp:.3f} (<0.05 = uneven error spread -> HC3 column);
Shapiro p = {sw:.3f} (residual skew {full.resid.skew():.2f}); max Cook's D = {cook:.3f}.
""")
    txt = "\n".join(out)
    (RES / "02_f_tests.md").write_text(txt, encoding="utf-8")
    print(txt)
