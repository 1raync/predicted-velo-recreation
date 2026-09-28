"""Figure: two athletes with the same predicted velo but opposite gaps
(the Athlete A / Athlete B idea from Driveline's Predicted Pitch Velocity blog).
Each athlete's prediction = average of out-of-fold predictions across the 10x5-fold CV,
so no athlete is ever predicted by a model that saw him. Outputs figures/two_athlete_gap.png"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from common import load, folds, FEATS_J as FEATS, TARGET, FIG, FOOT
from s01_baseline_model import cv_scores

df, _ = load()
OUT = FIG / "two_athlete_gap.png"
fs = folds(df)
preds = np.zeros(len(df)); counts = np.zeros(len(df))
for tr, te in fs:
    m = LinearRegression().fit(df.loc[tr, FEATS], df.loc[tr, TARGET])
    preds[te] += m.predict(df.loc[te, FEATS]); counts[te] += 1
df["pred"] = preds / counts
df["gap"] = df["pred"] - df[TARGET]          # + = body predicts more than he throws
r2s, maes = cv_scores(df, FEATS, fs)
r2, mae = r2s.mean(), maes.mean()
print(f"n={len(df)} athletes | CV R2={r2:.3f} | MAE={mae:.2f} mph")

# ---------- 4. pick the matched pair ----------
over = df[df["gap"].between(-10, -6)]        # throws harder than body predicts
under = df[df["gap"].between(6, 10)]         # body predicts more than he throws
pairs = over.merge(under, on="playing_level", suffixes=("_a", "_b"))
pairs["pred_diff"] = (pairs["pred_a"] - pairs["pred_b"]).abs()
best = pairs.sort_values("pred_diff").iloc[0]
A = df[df["athlete_uid"] == best["athlete_uid_a"]].iloc[0]
B = df[df["athlete_uid"] == best["athlete_uid_b"]].iloc[0]
level = A["playing_level"]

# percentiles within the same playing level
peers = df[df["playing_level"] == level]
PROFILE = [("SJ_PP", "SJ peak power"), ("IMTP_NPF", "IMTP net peak force"),
           ("Hop_RSI", "Hop RSI"), ("CMJ_RSImod", "CMJ RSI-mod"), ("BW_kg", "Bodyweight")]
pct = lambda col, v: (peers[col] < v).mean() * 100

# ---------- 5. plot ----------
CA, CB, GREY = "#1f77b4", "#d62728", "#b0b0b0"
fig = plt.figure(figsize=(14, 7.5))
gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1], height_ratios=[1, 1.15], hspace=0.45, wspace=0.28)

# left: everyone
ax = fig.add_subplot(gs[:, 0])
ax.scatter(df["pred"], df[TARGET], s=12, color=GREY, alpha=0.6, label="All athletes")
lo, hi = df[["pred", TARGET]].min().min() - 2, df[["pred", TARGET]].max().max() + 2
ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="Actual = predicted")
ax.fill_between([lo, hi], [lo - mae, hi - mae], [lo + mae, hi + mae], color="k", alpha=0.07,
                label=f"± average miss ({mae:.1f} mph)")
for r, c, tag in [(A, CA, "A"), (B, CB, "B")]:
    ax.scatter(r["pred"], r[TARGET], s=160, color=c, edgecolor="white", lw=2, zorder=5)
    ax.annotate(f"Athlete {tag}", (r["pred"], r[TARGET]), xytext=(10, 0),
                textcoords="offset points", color=c, fontweight="bold", va="center")
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_xlabel("Predicted velo from strength & jump tests (mph)")
ax.set_ylabel("Actual pitch velo (mph)")
ax.set_title(f"{len(df)} athletes, each predicted by a model that never saw them\n"
             f"held-out R² = {r2:.2f} (10×5-fold CV)", fontsize=11)
ax.legend(loc="upper left", fontsize=9, frameon=False)

# top right: predicted vs actual
ax2 = fig.add_subplot(gs[0, 1])
for y, r, c, tag, rx in [(1, A, CA, "A", "More weight room"), (0, B, CB, "B", "More throwing work")]:
    ax2.plot([r["pred"], r[TARGET]], [y, y], color=c, lw=3, alpha=0.5)
    ax2.scatter(r["pred"], y, s=120, facecolor="white", edgecolor=c, lw=2.5, zorder=3)
    ax2.scatter(r[TARGET], y, s=120, color=c, zorder=3)
    ax2.text(max(r["pred"], r[TARGET]) + 0.8, y,
             f"{r[TARGET] - r['pred']:+.1f} mph  →  {rx}", va="center", color=c, fontweight="bold")
ax2.set_yticks([1, 0], ["Athlete A", "Athlete B"])
ax2.set_ylim(-0.7, 1.7)
xmin = min(A[TARGET], B[TARGET], A["pred"], B["pred"]) - 3
ax2.set_xlim(xmin, xmin + 32)
ax2.set_xlabel("Pitch velo (mph)   ○ predicted   ● actual")
ax2.set_title(f"Same predicted velo (~{A['pred']:.0f} mph), {abs(A[TARGET] - B[TARGET]):.0f} mph apart on the mound",
              fontsize=11)
for s in ["top", "right"]:
    ax2.spines[s].set_visible(False)

# bottom right: percentile profile
ax3 = fig.add_subplot(gs[1, 1])
yy = np.arange(len(PROFILE))[::-1]
h = 0.36
ax3.barh(yy + h / 2, [pct(c, A[c]) for c, _ in PROFILE], h, color=CA, label="Athlete A")
ax3.barh(yy - h / 2, [pct(c, B[c]) for c, _ in PROFILE], h, color=CB, label="Athlete B")
ax3.axvline(50, color="k", lw=0.8, ls=":")
ax3.set_yticks(yy, [lab for _, lab in PROFILE])
ax3.set_xlim(0, 100)
ax3.set_xlabel(f"Percentile vs {len(peers)} {level} athletes")
ax3.set_title("Different strength profiles that add up to the same prediction", fontsize=11)
ax3.legend(loc="lower right", fontsize=9, frameon=False)
for s in ["top", "right"]:
    ax3.spines[s].set_visible(False)

fig.suptitle("Same predicted velo, different arm: where the gap points training",
             fontsize=15, fontweight="bold", x=0.02, ha="left")
fig.text(0.02, 0.005, FOOT, fontsize=8, color="#555")
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print("saved", OUT)
print(f"A: {level}, pred {A['pred']:.1f}, actual {A[TARGET]:.1f}")
print(f"B: {level}, pred {B['pred']:.1f}, actual {B[TARGET]:.1f}")
