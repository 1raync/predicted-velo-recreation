"""Figures: feature impact (permutation importance) and the bodyweight curve.
Outputs figures/feature_impact.png and figures/bodyweight_curve.png"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from common import load, folds, add_bw_terms, BASE, FEATS_J, TARGET, FIG, FOOT, SEED

df, _ = load()
BW_MEAN = df["BW_kg"].mean()
y = df[TARGET].values
folds = folds(df)
print(f"n = {len(df)} athletes, {len(folds)} fits")

# ================= Figure 1: feature impact =================
GROUPS = {  # label -> raw column(s) to shuffle together
    "SJ peak power": ["SJ_PP"],
    "IMTP net peak force": ["IMTP_NPF"],
    "Bodyweight (curve)": ["BW_kg"],
    "Hop RSI": ["Hop_RSI"],
    "CMJ RSI-mod": ["CMJ_RSImod"],
}
rng = np.random.default_rng(SEED)
imp = {g: [] for g in GROUPS}
for tr, te in folds:
    m = LinearRegression().fit(df.loc[tr, FEATS_J], y[tr])
    test = df.loc[te]
    base_r2 = r2_score(y[te], m.predict(test[FEATS_J]))
    for g, cols in GROUPS.items():
        drops = []
        for _ in range(10):
            sh = test.copy()
            perm = rng.permutation(len(sh))
            sh[cols] = sh[cols].values[perm]
            sh = add_bw_terms(sh, BW_MEAN)            # rebuild BW terms if BW was shuffled
            drops.append(base_r2 - r2_score(y[te], m.predict(sh[FEATS_J])))
        imp[g].append(np.mean(drops))
imp = pd.DataFrame(imp)
ranks = imp.rank(axis=1, ascending=False)
summary = pd.DataFrame({
    "mean": imp.mean(), "p5": imp.quantile(0.05), "p95": imp.quantile(0.95),
    "rank1_share": (ranks == 1).mean(), "modal_rank": ranks.mode().iloc[0],
}).sort_values("mean")
print(summary.round(3))

DRIVELINE_ORDER = {"SJ peak power": "1st", "CMJ RSI-mod": "2nd", "Hop RSI": "3rd",
                   "IMTP net peak force": "4th", "Bodyweight (curve)": "not used"}
fig, ax = plt.subplots(figsize=(10, 5.2))
colors = ["#1f77b4" if g != "Bodyweight (curve)" else "#ff7f0e" for g in summary.index]
yy = np.arange(len(summary))
ax.barh(yy, summary["mean"], color=colors, height=0.6)
ax.errorbar(summary["mean"], yy,
            xerr=[summary["mean"] - summary["p5"], summary["p95"] - summary["mean"]],
            fmt="none", ecolor="k", capsize=4, lw=1)
labels = [f"{g}\n(Driveline list: {DRIVELINE_ORDER[g]})" for g in summary.index]
ax.set_yticks(yy, labels)
for i, (g, r) in enumerate(summary.iterrows()):
    ax.text(r["p95"] + 0.006, i, f"{r['mean']:.3f}", va="center", fontsize=10)
ax.set_xlabel("Drop in held-out R² when this input is shuffled (bigger = model relies on it more)")
ax.set_xlim(0, summary["p95"].max() * 1.18)
top = summary.index[-1]
ax.set_title(f"Feature impact: {top} dominates (ranked #1 in "
             f"{summary.loc[top, 'rank1_share']:.0%} of {len(folds)} fits)",
             loc="left", fontsize=13, fontweight="bold")
ax.text(0, -0.2, "Model: Driveline's 4 inputs + bodyweight curve.  Whiskers = 5th–95th percentile across fits.  "
        "Blue = Driveline input, orange = added.", transform=ax.transAxes, fontsize=9, color="#333")
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
fig.text(0.01, -0.08, FOOT, fontsize=8, color="#555")
fig.savefig(FIG / "feature_impact.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ================= Figure 2: bodyweight curve =================
mJ = LinearRegression().fit(df[FEATS_J], y)
b1, b2 = mJ.coef_[FEATS_J.index("BW_c")], mJ.coef_[FEATS_J.index("BW_c2")]
bw_effect = lambda kg: b1 * (kg - BW_MEAN) + b2 * (kg - BW_MEAN) ** 2
partial_resid = y - mJ.predict(df[FEATS_J]) + bw_effect(df["BW_kg"].values)
peak_kg = BW_MEAN - b1 / (2 * b2)

grid_kg = np.linspace(df["BW_kg"].quantile(0.01), df["BW_kg"].quantile(0.99), 200)
boot = []
for _ in range(1000):
    s = df.sample(len(df), replace=True, random_state=rng.integers(1e9))
    mb = LinearRegression().fit(s[FEATS_J], s[TARGET])
    c1, c2 = mb.coef_[FEATS_J.index("BW_c")], mb.coef_[FEATS_J.index("BW_c2")]
    boot.append(c1 * (grid_kg - BW_MEAN) + c2 * (grid_kg - BW_MEAN) ** 2)
lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
p95_kg = df["BW_kg"].quantile(0.95)

# held-out R2 for ways of using bodyweight
VARIANTS = {
    "All per-kg\n(SJ/kg, IMTP/kg)": ["SJ_PP_perkg", "CMJ_RSImod", "Hop_RSI", "IMTP_perkg"],
    "SJ → SJ/kg": ["SJ_PP_perkg", "CMJ_RSImod", "Hop_RSI", "IMTP_NPF"],
    "Driveline 4\n(baseline)": BASE,
    "+ BW\n(line)": BASE + ["BW_c"],
    "+ BW\n(curve)": FEATS_J,
}
var_r2 = {}
for k, f in VARIANTS.items():
    var_r2[k] = np.mean([r2_score(y[te], LinearRegression().fit(df.loc[tr, f], y[tr]).predict(df.loc[te, f]))
                         for tr, te in folds])
print({k.replace("\n", " ").replace("→", "->"): round(v, 3) for k, v in var_r2.items()})  # "→" crashes cp1252 consoles
print(f"curve peak {peak_kg:.1f} kg ({peak_kg * 2.2046:.0f} lb); 95th pct BW {p95_kg * 2.2046:.0f} lb")

fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.5, 1], "wspace": 0.3})
lb = lambda kg: kg * 2.2046
a1.scatter(lb(df["BW_kg"]), partial_resid, s=12, color="#b0b0b0", alpha=0.6,
           label="Athletes (velo left after strength/jump inputs)")
a1.fill_between(lb(grid_kg), lo, hi, color="#ff7f0e", alpha=0.2, label="95% bootstrap band")
a1.plot(lb(grid_kg), bw_effect(grid_kg), color="#ff7f0e", lw=3, label="Fitted bodyweight curve")
a1.axhline(0, color="k", lw=0.8, ls=":")
a1.axvspan(lb(p95_kg), df["BW_lb"].max() + 5, color="k", alpha=0.06)
a1.text(lb(p95_kg) + 1, partial_resid.max(),
        "top 5% heaviest:\nfew athletes,\nwide band", fontsize=9, color="#444", va="top")
a1.axvline(lb(peak_kg), color="#ff7f0e", lw=1, ls="--")
a1.annotate(f"curve peaks ≈ {lb(peak_kg):.0f} lb", (lb(peak_kg), bw_effect(peak_kg)),
            xytext=(-130, 50), textcoords="offset points", color="#c55a00", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#c55a00", alpha=0.9),
            arrowprops=dict(arrowstyle="->", color="#c55a00"))
a1.set_xlim(df["BW_lb"].min() - 5, df["BW_lb"].max() + 5)   # show every athlete
pad = 1.0
a1.set_ylim(partial_resid.min() - pad, partial_resid.max() + pad)   # show every athlete
a1.set_xlabel("Bodyweight (lb)")
a1.set_ylabel(f"Velo effect vs a {lb(BW_MEAN):.0f}-lb athlete\nwith the same strength scores (mph)")
a1.set_title("(a) At equal strength, heavier throws harder, then levels off", loc="left", fontsize=12)
a1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=9, frameon=False)

keys = list(VARIANTS)
vals = [var_r2[k] for k in keys]
cols = ["#d62728", "#d62728", "#7f7f7f", "#ffbb78", "#ff7f0e"]
a2.bar(range(len(keys)), vals, color=cols)
for i, v in enumerate(vals):
    a2.text(i, v + 0.008, f"{v:.2f}", ha="center", fontsize=10)
a2.set_xticks(range(len(keys)), keys, fontsize=9)
a2.set_ylim(0, max(vals) * 1.2)
a2.set_ylabel("Held-out R² (higher = better)")
a2.set_title("(b) Dividing by bodyweight hurts;\nadding it as a curve helps", loc="left", fontsize=12)
for ax in (a1, a2):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
fig.suptitle("How bodyweight fits into predicted velo", x=0.01, ha="left", fontsize=15, fontweight="bold")
fig.text(0.01, -0.1, FOOT + f"  n = {len(df)} athletes; R² from 10×5-fold CV.", fontsize=8, color="#555")
fig.savefig(FIG / "bodyweight_curve.png", dpi=200, bbox_inches="tight")
print("saved feature_impact.png, bodyweight_curve.png")
