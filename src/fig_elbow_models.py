"""Figure: two matched pitchers, one with high and one with low elbow stress beyond expected, each drawn at max
layback. Reads data/derived/elbow_pair_* written by s05_elbow_stress.py; renders via render_pitcher.py (pyvista).
Outputs figures/elbow_models.png"""
import pandas as pd
import matplotlib.pyplot as plt
from common import ROOT, FIG
from render_pitcher import render
from fig_lead_leg import COL

DER = ROOT / "data" / "derived"

if __name__ == "__main__":
    info = pd.read_csv(DER / "elbow_pair_info.csv").set_index("tag")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.6))
    for ax, tag in zip(axes, ["H", "L"]):
        r, c = info.loc[tag], COL[tag]
        ax.imshow(render(tag, c, pair="elbow_pair", force=False, arm=True))
        ax.axis("off")
        ax.set_title(f"{r.velo:.1f} mph  ·  {r.height_m:.2f} m  ·  {r.mass_kg:.1f} kg\n"
                     f"(max layback, {abs(r.frame_ms_vs_BR):.0f} ms before release)", fontsize=11.5, color=c,
                     fontweight="bold")
        ax.text(0.04, 0.96, f"Elbow varus {r.varus_nm:.0f} Nm ({r.resid_nm:+.0f} vs expected)\n"
                f"Max layback {r.layback_deg:.0f}°", transform=ax.transAxes, va="top", color="white", fontsize=12,
                fontweight="bold", bbox=dict(boxstyle="round,pad=0.4", fc=c, ec="none", alpha=0.9))
    fig.subplots_adjust(wspace=0.04)
    fig.savefig(FIG / "elbow_models.png", dpi=200, bbox_inches="tight")
    print("saved", FIG / "elbow_models.png")
