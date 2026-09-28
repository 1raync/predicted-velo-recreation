"""3D render of one pitcher at one frame, in the style of a marker-based motion-capture viewer: grey scene,
floor grid, lab axis triad, force plates, a stick skeleton through the joint centres, raw markers, and the
ground-reaction-force arrow. Plotting only: reads the data/derived/pitch_pair_* files written by
s04_lead_leg_block.py. Uses pyvista (VTK), rendered off-screen.
Joint centres are OBP's (Visual3D model); marker positions are the raw C3D markers."""
import numpy as np
import pandas as pd
import pyvista as pv
from common import ROOT

DER = ROOT / "data" / "derived"
M_PER_BW = 0.3                       # force arrow: 0.3 m of length per bodyweight
BONE, MARKER = "#f3e9a8", "#e6e6e6"
BG_BOTTOM, BG_TOP, FLOOR, GRID = "#3a3d41", "#707377", "#46494d", "#a3a6aa"
BONE_R, JOINT_R, MARKER_R = 0.009, 0.022, 0.011
# stick skeleton: joint-centre names are OBP landmark columns without _x/_y/_z; RTOE etc. are raw markers
BONES = [("RTOE", "rear_ankle_jc"), ("RHEE", "rear_ankle_jc"), ("RHEE", "RTOE"),
         ("rear_ankle_jc", "rear_knee_jc"), ("rear_knee_jc", "rear_hip"),
         ("LTOE", "lead_ankle_jc"), ("LHEE", "lead_ankle_jc"), ("LHEE", "LTOE"),
         ("lead_ankle_jc", "lead_knee_jc"), ("lead_knee_jc", "lead_hip"),
         ("rear_hip", "lead_hip"), ("rear_hip", "thorax_dist"), ("lead_hip", "thorax_dist"),
         ("thorax_dist", "thorax_prox"), ("thorax_prox", "shoulder_jc"), ("thorax_prox", "glove_shoulder_jc"),
         ("shoulder_jc", "elbow_jc"), ("elbow_jc", "wrist_jc"), ("wrist_jc", "hand_jc"),
         ("glove_shoulder_jc", "glove_elbow_jc"), ("glove_elbow_jc", "glove_wrist_jc"),
         ("glove_wrist_jc", "glove_hand_jc"), ("thorax_prox", "head"),
         ("LFHD", "RFHD"), ("RFHD", "RBHD"), ("RBHD", "LBHD"), ("LBHD", "LFHD")]
JOINTS = ["rear_ankle_jc", "rear_knee_jc", "rear_hip", "lead_ankle_jc", "lead_knee_jc", "lead_hip", "thorax_dist",
          "thorax_prox", "shoulder_jc", "elbow_jc", "wrist_jc", "hand_jc", "glove_shoulder_jc", "glove_elbow_jc",
          "glove_wrist_jc", "glove_hand_jc", "head"]
# same view for every pitcher, from the 3B side: the plane in which the force angle is measured
CAMERA = [(1.9, -4.7, 2.4), (1.15, -0.3, 0.72), (0, 0, 1)]


def points(frame, mk):
    """Joint centres (m, lab frame) plus raw markers, keyed by name."""
    names = {c[:-2] for c in frame.index if c.endswith("_x")}
    J = {n: np.array([frame[f"{n}_x"], frame[f"{n}_y"], frame[f"{n}_z"]], float) for n in names}
    J |= {m: r[["x", "y", "z"]].values.astype(float) for m, r in mk.iterrows()}
    J["head"] = mk.loc[["LFHD", "RFHD", "LBHD", "RBHD"], ["x", "y", "z"]].mean().values   # head-band centroid
    return J


def arrow(p, start, vec, length, color):
    p.add_mesh(pv.Arrow(start=start, direction=vec, scale=length, tip_length=0.2, tip_radius=0.08,
                        shaft_radius=0.03), color=color, ambient=0.4, smooth_shading=True)


def render(tag, accent, size=(1100, 1100)):
    info = pd.read_csv(DER / "pitch_pair_info.csv").set_index("tag").loc[tag]
    frame = pd.read_csv(DER / "pitch_pair_frames.csv").set_index("tag").loc[tag]
    mk = pd.read_csv(DER / "pitch_pair_markers.csv").query("tag == @tag").set_index("label")
    plates = pd.read_csv(DER / "pitch_pair_plates.csv").query("tag == @tag")
    J = points(frame, mk)

    p = pv.Plotter(off_screen=True, window_size=size)
    p.set_background(BG_BOTTOM, top=BG_TOP)
    floor_z = plates.z.min() - 0.01      # below every plate corner: the landing plates sit ~8 cm down the mound slope
    p.add_mesh(pv.Plane(center=(1.0, 2.0, floor_z), i_size=16, j_size=16), color=FLOOR, ambient=0.6)
    p.add_mesh(pv.Plane(center=(1.0, 2.0, floor_z + 0.002), i_size=16, j_size=16, i_resolution=64, j_resolution=64),
               style="wireframe", color=GRID, line_width=1, opacity=0.6)   # 0.25 m grid
    for _, pl in plates.groupby("plate"):
        pts = pl[["x", "y", "z"]].values
        if pl.loaded.iloc[0]:
            p.add_mesh(pv.PolyData(pts, faces=[4, 0, 1, 2, 3]), color=accent, opacity=0.35)
        p.add_mesh(pv.lines_from_points(np.vstack([pts, pts[:1]])), color="#d0d4d8", line_width=2)
    for v, c in [((1, 0, 0), "#e8322c"), ((0, 1, 0), "#2fbf3a"), ((0, 0, 1), "#2f6bff")]:   # lab axes at origin
        arrow(p, np.zeros(3), v, 0.35, c)

    for a, b in BONES:
        p.add_mesh(pv.Tube(pointa=J[a], pointb=J[b], radius=BONE_R, n_sides=12), color=BONE, ambient=0.3,
                   smooth_shading=True)
    for n in JOINTS:
        p.add_mesh(pv.Sphere(radius=JOINT_R, center=J[n]), color=accent, ambient=0.3, smooth_shading=True)
    for m in mk.index:
        p.add_mesh(pv.Sphere(radius=MARKER_R, center=J[m]), color=MARKER, ambient=0.5, smooth_shading=True)

    cop = info[["cop_x", "cop_y", "cop_z"]].values.astype(float)
    vec = info[["gx_bw", "gy_bw", "gz_bw"]].values.astype(float)
    arrow(p, cop, vec, np.linalg.norm(vec) * M_PER_BW, accent)
    ref = np.array([cop[0] + 0.55, cop[1] + 0.45, floor_z])      # 1x bodyweight reference arrow, vertical
    arrow(p, ref, (0, 0, 1), M_PER_BW, "#c9ccd0")
    p.camera_position = CAMERA
    p.camera.view_angle = 25
    p.enable_anti_aliasing("ssaa")
    img = p.screenshot(return_img=True)
    p.close()
    return img
