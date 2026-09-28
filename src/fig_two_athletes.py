"""Headline figure: two athletes who throw the same velo but sit on opposite sides of their
predicted velo (the Athlete A / Athlete B idea from Driveline's Predicted Pitch Velocity blog).
Each athlete's prediction = average of out-of-fold predictions across the 10x5-fold CV,
so no athlete is ever predicted by a model that saw them.

Pair rule (same playing level, actual velo within VELO_TOL mph):
  A: predicted velo at or below the level median, throws >= GAP mph ABOVE prediction
  B: predicted velo in the level's top 25%,       throws >= GAP mph BELOW prediction
  ties -> closest actual velo, then largest difference in predicted velo.
DECIDED: match on actual velo, not predicted velo (the earlier version). A same-prediction pair didn't
  guarantee a strong B; same velo makes "strong but throws below" vs "not strong, throws above" explicit.
DECIDED: "strength" = the model's predicted velo, ranked within playing level. Rejected ranking on a single test
  (e.g. SJ power) because the gap is measured against the prediction.
compute() writes data/derived/two_athletes_*.csv; plot() reads them. Outputs figures/two_athlete_gap.png"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from common import ROOT, load, folds, FEATS_J as FEATS, TARGET, FIG, FOOT
from s01_baseline_model import cv_scores

DER = ROOT / "data" / "derived"
GAP = 6          # mph, ~1.5x the final model's held-out MAE (3.8 mph)
VELO_TOL = 1     # mph
PROFILE = [("SJ_PP", "SJ peak power", "{:.0f} W"), ("IMTP_NPF", "IMTP net peak force", "{:.0f} N"),
           ("CMJ_RSImod", "CMJ RSI-mod", "{:.2f}"), ("Hop_RSI", "Hop RSI", "{:.2f}"),
           ("BW_lb", "Bodyweight", "{:.0f} lb")]


def compute():
    DER.mkdir(parents=True, exist_ok=True)
    df, _ = load()
    fs = folds(df)
    preds, counts = np.zeros(len(df)), np.zeros(len(df))
    for tr, te in fs:
        m = LinearRegression().fit(df.loc[tr, FEATS], df.loc[tr, TARGET])
        preds[te] += m.predict(df.loc[te, FEATS]); counts[te] += 1
    df["pred"] = preds / counts
    df["gap"] = df[TARGET] - df["pred"]          # + = throws harder than the tests predict
    df["pred_pct"] = df.groupby("playing_level")["pred"].rank(pct=True)
    r2s, maes = cv_scores(df, FEATS, fs)

    a = df[(df.pred_pct <= 0.5) & (df.gap >= GAP)]
    b = df[(df.pred_pct >= 0.75) & (df.gap <= -GAP)]
    pairs = a.merge(b, on="playing_level", suffixes=("_a", "_b"))
    pairs["dv"] = (pairs[TARGET + "_a"] - pairs[TARGET + "_b"]).abs()
    pairs["dp"] = pairs["pred_b"] - pairs["pred_a"]
    pairs = pairs[pairs.dv <= VELO_TOL].sort_values(["dv", "dp"], ascending=[True, False])
    print(f"n={len(df)} | candidates A={len(a)}, B={len(b)} | same-level pairs within {VELO_TOL} mph: {len(pairs)}")
    best = pairs.iloc[0]
    df["who"] = df.athlete_uid.map({best.athlete_uid_a: "A", best.athlete_uid_b: "B"})

    peers = df[df.playing_level == best.playing_level]
    prof = []
    for col, lab, fmt in PROFILE:
        for who in "AB":
            v = df.loc[df.who == who, col].iloc[0]
            prof.append([who, lab, fmt.format(v), (peers[col] < v).mean() * 100])
    df[["pred", TARGET, "who"]].to_csv(DER / "two_athletes_all.csv", index=False)
    pd.DataFrame(prof, columns=["who", "test", "value", "pct"]).to_csv(DER / "two_athletes_profile.csv", index=False)
    pd.Series({"r2": r2s.mean(), "mae": maes.mean(), "level": best.playing_level, "n_peers": len(peers),
               "n_pairs": len(pairs)}).to_csv(DER / "two_athletes_info.csv")
    for who in "AB":
        r = df[df.who == who].iloc[0]
        print(f"{who}: {r.playing_level}, pred {r.pred:.1f}, actual {r[TARGET]:.1f}, gap {r.gap:+.1f}, "
              f"pred pct {r.pred_pct:.2f}")


def plot():
    df = pd.read_csv(DER / "two_athletes_all.csv")
    prof = pd.read_csv(DER / "two_athletes_profile.csv")
    info = pd.read_csv(DER / "two_athletes_info.csv", index_col=0).iloc[:, 0]
    r2, mae = float(info.r2), float(info.mae)
    A, B = df[df.who == "A"].iloc[0], df[df.who == "B"].iloc[0]
    CA, CB, GREY = "#1f77b4", "#d62728", "#b0b0b0"
    fig = plt.figure(figsize=(14, 7.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1], height_ratios=[1, 1.25], hspace=0.5, wspace=0.28)

    # left: everyone
    ax = fig.add_subplot(gs[:, 0])
    ax.scatter(df["pred"], df[TARGET], s=12, color=GREY, alpha=0.6, label="All athletes")
    lo, hi = df[["pred", TARGET]].min().min() - 2, df[["pred", TARGET]].max().max() + 2
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="Actual = predicted")
    ax.fill_between([lo, hi], [lo - mae, hi - mae], [lo + mae, hi + mae], color="k", alpha=0.07,
                    label=f"± average miss ({mae:.1f} mph)")
    ax.plot([A["pred"], B["pred"]], [A[TARGET], B[TARGET]], color="k", lw=1, ls=":", zorder=4)
    ax.text((A["pred"] + B["pred"]) / 2, A[TARGET] - 1.6, f"same velo,\n{B['pred'] - A['pred']:.0f} mph apart in prediction",
            ha="center", va="top", fontsize=9, bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.9),
            zorder=6)
    for r, c, tag in [(A, CA, "A"), (B, CB, "B")]:
        ax.scatter(r["pred"], r[TARGET], s=160, color=c, edgecolor="white", lw=2, zorder=5)
        ax.annotate(f"Athlete {tag}", (r["pred"], r[TARGET]), xytext=(0, 12), ha="center",
                    textcoords="offset points", color=c, fontweight="bold")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Predicted velo from strength & jump tests + bodyweight (mph)")
    ax.set_ylabel("Actual pitch velo (mph)")
    ax.set_title(f"{len(df)} athletes, each predicted by a model that never saw them\n"
                 f"held-out R² = {r2:.2f} (10×5-fold CV)", fontsize=11)
    ax.legend(loc="upper left", fontsize=9, frameon=False)

    # top right: predicted vs actual
    ax2 = fig.add_subplot(gs[0, 1])
    rows = [(1, A, CA, "A", "more weight room"),
            (0, B, CB, "B", "more throwing & mechanics work")]
    for y, r, c, tag, rx in rows:
        ax2.plot([r["pred"], r[TARGET]], [y, y], color=c, lw=3, alpha=0.5)
        ax2.scatter(r["pred"], y, s=120, facecolor="white", edgecolor=c, lw=2.5, zorder=3)
        ax2.scatter(r[TARGET], y, s=120, color=c, zorder=3)
        ax2.text((r["pred"] + r[TARGET]) / 2, y + 0.3, f"{r[TARGET] - r['pred']:+.1f} mph vs predicted → {rx}",
                 ha="center", color=c, fontweight="bold", fontsize=9.5)
    ax2.set_yticks([1, 0], ["Athlete A", "Athlete B"])
    ax2.set_ylim(-0.5, 1.8)
    xs = [A["pred"], B["pred"], A[TARGET], B[TARGET]]
    ax2.set_xlim(min(xs) - 6, max(xs) + 6)
    ax2.set_xlabel("Pitch velo (mph)   ○ predicted   ● actual")
    ax2.set_title(f"Both throw ~{A[TARGET]:.0f} mph; their tests predict {A['pred']:.0f} vs {B['pred']:.0f} mph",
                  fontsize=11)
    for s in ["top", "right"]:
        ax2.spines[s].set_visible(False)

    # bottom right: force-plate test profile (percentile within the same playing level)
    ax3 = fig.add_subplot(gs[1, 1])
    tests = list(dict.fromkeys(prof.test))
    yy = np.arange(len(tests))[::-1]
    h = 0.38
    for who, c, off in [("A", CA, h / 2), ("B", CB, -h / 2)]:
        p = prof[prof.who == who].set_index("test").loc[tests]
        ax3.barh(yy + off, p.pct, h, color=c, label=f"Athlete {who}")
        for y, (pct, val) in zip(yy + off, zip(p.pct, p.value)):
            ax3.text(pct + 1, y, val, va="center", fontsize=8, color="#333")
    ax3.axvline(50, color="k", lw=0.8, ls=":")
    ax3.set_yticks(yy, tests)
    ax3.set_xlim(0, 112)
    ax3.set_xticks([0, 25, 50, 75, 100])
    ax3.set_xlabel(f"Percentile vs {int(info.n_peers)} {info.level} athletes (label = raw value)")
    pw = prof[prof.test != "Bodyweight"].pivot(index="test", columns="who", values="pct")
    ax3.set_title(f"Force-plate tests: B out-tests A on {(pw.B > pw.A).sum()} of {len(pw)}", fontsize=11)
    ax3.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=9, frameon=False)
    for s in ["top", "right"]:
        ax3.spines[s].set_visible(False)

    fig.suptitle("Same velo, opposite gaps: where the gap points training",
                 fontsize=15, fontweight="bold", x=0.02, ha="left")
    fig.text(0.02, 0.005, FOOT, fontsize=8, color="#555")
    fig.savefig(FIG / "two_athlete_gap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", FIG / "two_athlete_gap.png")


if __name__ == "__main__":
    compute()
    plot()
