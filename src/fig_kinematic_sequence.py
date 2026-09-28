"""Optional figure: kinematic sequence (segment rotation speeds over time) for the same two
OBP pitchers as fig_lead_leg.py. Reads data/derived/pitch_pair_*.csv written by s04_lead_leg_block.py.
Signals are OBP's joint angular velocities (deg/s, 360 Hz), plotted as provided (no filtering).
Outputs figures/kinematic_sequence.png"""
import pandas as pd
import matplotlib.pyplot as plt
from common import ROOT, FIG, FOOT

DER = ROOT / "data" / "derived"
# column -> label; columns verified to reproduce OBP's POI peak values exactly
SEQ = {"pelvis_velo_z": ("Pelvis rotation", "#8ecae6"), "torso_velo_z": ("Torso rotation", "#219ebc"),
       "elbow_velo_x": ("Elbow extension", "#fb8500"), "shoulder_velo_z": ("Shoulder internal rotation", "#d62828")}
COL = {"H": "#2a9d8f", "L": "#9b5de5"}

if __name__ == "__main__":
    v = pd.read_csv(DER / "pitch_pair_velos.csv")
    info = pd.read_csv(DER / "pitch_pair_info.csv").set_index("tag")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
    for ax, tag in zip(axes, ["H", "L"]):
        g = v[(v.tag == tag) & v.t_ms.between(-200, 60)]
        for c, (lab, colr) in SEQ.items():
            ax.plot(g.t_ms, g[c], color=colr, lw=2, label=lab)
            pk = g.loc[g[c].idxmax()]
            ax.scatter(pk.t_ms, pk[c], color=colr, s=30, zorder=3)
            ax.annotate(f"{pk[c]:.0f}", (pk.t_ms, pk[c]), xytext=(4, 4), textcoords="offset points", fontsize=8, color=colr)
        ax.axvline(0, color="k", lw=0.8, ls=":")
        ax.set_xlabel("ms relative to ball release")
        ax.set_title(f"{info.loc[tag, 'velo']:.1f} mph  ·  {info.loc[tag, 'height_m']:.2f} m  ·  "
                     f"{info.loc[tag, 'mass_kg']:.1f} kg", color=COL[tag],
                     fontweight="bold", fontsize=12)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("Angular velocity (deg/s)")
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle("Kinematic sequence: pelvis → torso → arm, same two pitchers (dots = peaks)",
                 x=0.02, ha="left", fontsize=14, fontweight="bold")
    fig.text(0.02, 0.9, "OBP pitching-biomechanics dataset; different athletes from the force-plate tests.",
             fontsize=9, color="#444")
    fig.text(0.02, -0.03, FOOT, fontsize=8, color="#555")
    fig.subplots_adjust(top=0.82)
    fig.savefig(FIG / "kinematic_sequence.png", dpi=200, bbox_inches="tight")
    print("saved", FIG / "kinematic_sequence.png")
