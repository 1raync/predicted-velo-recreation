"""Figure: the lead-leg block, from OBP's pitching biomechanics dataset (different athletes from
the force-plate tests). Reads data/derived/pitch_* written by s04_lead_leg_block.py; the 3D panels are
rendered by render_pitcher.py (pyvista).
Outputs figures/lead_leg_block.png (renders + force-time + cohort) and figures/lead_leg_models.png (renders only)"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from common import ROOT, FIG, FOOT
from render_pitcher import render

DER = ROOT / "data" / "derived"
COL = {"H": "#2a9d8f", "L": "#9b5de5"}
KEY = ("Coloured arrow: ground push on the front foot, from the measured centre of pressure.  Light-grey arrow: "
       "1× bodyweight for scale.\nColoured dots: joint centres from OBP's biomechanical model.  White dots: raw C3D "
       "markers.  Red/green/blue: lab x (toward home) / y / z axes.")


def pitcher_panel(ax, img, r, c):
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(f"{r.velo:.1f} mph  ·  {r.height_m:.2f} m  ·  {r.mass_kg:.1f} kg\n"
                 f"(peak force, {abs(r.peak_ms_vs_BR):.0f} ms before release)", fontsize=11.5, color=c, fontweight="bold")
    ax.text(0.04, 0.96, f"{r.peak_bw:.2f}× bodyweight\n{r.angle_deg:.0f}° above horizontal", transform=ax.transAxes,
            va="top", color="white", fontsize=12, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", fc=c, ec="none", alpha=0.9))


if __name__ == "__main__":
    info = pd.read_csv(DER / "pitch_pair_info.csv").set_index("tag")
    grf = pd.read_csv(DER / "pitch_pair_grf.csv")
    coh = pd.read_csv(DER / "pitchers_cohort.csv")
    cst = pd.read_csv(DER / "pitchers_cohort_stats.csv", index_col=0).iloc[:, 0]
    imgs = {tag: render(tag, COL[tag]) for tag in ["H", "L"]}

    # renders only (README, under the headline figure)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.6))
    for ax, tag in zip(axes, ["H", "L"]):
        pitcher_panel(ax, imgs[tag], info.loc[tag], COL[tag])
    fig.subplots_adjust(wspace=0.04)
    fig.savefig(FIG / "lead_leg_models.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", FIG / "lead_leg_models.png")

    fig = plt.figure(figsize=(17, 8))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.05], hspace=0.5, wspace=0.28)
    for col, tag in enumerate(["H", "L"]):
        pitcher_panel(fig.add_subplot(gs[:, col]), imgs[tag], info.loc[tag], COL[tag])
    fig.text(0.02, 0.1, KEY, fontsize=8.5, color="#444", va="top")

    # force over time
    a1 = fig.add_subplot(gs[0, 2])
    for tag in ["H", "L"]:
        g = grf[(grf.tag == tag) & grf.t_ms.between(-250, 100)]
        a1.plot(g.t_ms, g.mag_bw, color=COL[tag], lw=2, label=f"{info.loc[tag, 'velo']:.0f} mph")
        pk = g.loc[g.mag_bw.idxmax()]
        a1.scatter(pk.t_ms, pk.mag_bw, color=COL[tag], s=40, zorder=3)   # = the frame rendered at left
    a1.axvline(0, color="k", lw=0.8, ls=":")
    a1.text(2, a1.get_ylim()[1] * 0.97, "ball release", fontsize=8, va="top")
    a1.set_xlabel("ms relative to ball release")
    a1.set_ylabel("Lead-leg force (× bodyweight)")
    a1.set_title("Lead-leg ground reaction force (dot = frame shown)", fontsize=11, loc="left")
    a1.legend(frameon=False, fontsize=9, loc="upper left")

    # cohort
    a2 = fig.add_subplot(gs[1, 2])
    a2.scatter(coh.lead_bw, coh.velo, s=14, color="#b0b0b0", alpha=0.8)
    b = np.polyfit(coh.lead_bw, coh.velo, 1)
    xs = np.linspace(coh.lead_bw.min(), coh.lead_bw.max(), 2)
    a2.plot(xs, np.polyval(b, xs), color="k", lw=1)
    for tag in ["H", "L"]:
        r = coh[coh.pair == tag].iloc[0]
        a2.scatter(r.lead_bw, r.velo, s=90, color=COL[tag], edgecolor="white", lw=1.5, zorder=3)
    a2.set_xlabel("Peak lead-leg force (× bodyweight; pitcher mean)")
    a2.set_ylabel("Pitch velo (mph)")
    a2.set_title(f"All {int(cst.n)} OBP pitchers: r = {float(cst.r):.2f} (95% CI {cst.ci.replace(' to ', '–')})",
                 fontsize=11, loc="left")
    for ax in (a1, a2):
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)

    dv = info.loc["H", "velo"] - info.loc["L", "velo"]
    fig.suptitle(f"The lead-leg block: two {info.loc['H', 'level']} right-handers, same height, {dv:.0f} mph apart",
                 x=0.02, ha="left", fontsize=15, fontweight="bold")
    fig.text(0.02, 0.925, "Illustration from OBP's separate pitching-biomechanics dataset (not the athletes above; OBP "
             "does not link the two). 3D view from the third-base side, rendered in Python from OBP's C3D files.",
             fontsize=9, color="#444")
    fig.text(0.02, 0.02, FOOT, fontsize=8, color="#555")
    fig.savefig(FIG / "lead_leg_block.png", dpi=200, bbox_inches="tight")
    print("saved", FIG / "lead_leg_block.png")
