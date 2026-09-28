"""Step 4: lead-leg block in OBP's *pitching biomechanics* dataset (a different set of athletes
from the force-plate tests: OBP provides no crosswalk between the two).
  1. Cohort check: does peak lead-leg ground reaction force (GRF) track velo? One row per pitcher.
  2. Pick an illustrative pair: right-handers, same level, body mass within 5 kg, height within 5 cm,
     largest velo gap, each represented by the pitch closest to that pitcher's median velo.
     GRF is NOT a selection criterion.
  3. Save their force, joint-position and joint-velocity signals, plus the raw C3D markers, force-plate
     corners and centre of pressure (COP) at the drawn frame, for the figures.
Outputs results/04_lead_leg_block.md and data/derived/pitch_*.csv

DECIDED: one row per pitcher for the cohort r. Rejected per-pitch rows: ~4 pitches per pitcher are not
  independent and would overstate certainty.
DECIDED: headline force in bodyweights (heavier pitchers push harder AND throw harder); raw newtons with a
  mass-controlled partial r is reported alongside rather than instead.
DECIDED: draw each pitcher at peak lead-leg force, not at ball release. The peak is the cohort metric, and at
  release the ranking can flip (a faster pitcher's force may already have dropped off).
DECIDED: match height (within 5 cm) as well as mass. Mass alone paired a 1.85 m pitcher with a 1.70 m one,
  and height is itself a velo factor. The cost is a smaller force gap in the drawn pair."""
import zipfile
import ezc3d
import numpy as np
import pandas as pd
from scipy import stats
from common import ROOT, RES, SEED, md_table

PIT = ROOT / "data" / "pitching"
DER = ROOT / "data" / "derived"
G = 9.81
MASS_TOL_KG = 5
HEIGHT_TOL_M = 0.05
LANDING_X_M = 1.0   # plates centred beyond this x are the landing plates (the rubber plate sits near x = 0)


def read_zip(name, **kw):
    z = zipfile.ZipFile(PIT / f"{name}.zip")
    return pd.read_csv(z.open(z.namelist()[0]), **kw)


def boot_r(x, y, rng, n=2000):
    idx = (rng.integers(0, len(x), len(x)) for _ in range(n))
    return np.percentile([np.corrcoef(x[i], y[i])[0, 1] for i in idx], [2.5, 97.5])


def read_c3d(fname):
    """Extract one pitch's C3D from the release zip (if needed) and read it with its force platforms."""
    path = PIT / "c3d" / fname[:6] / fname
    if not path.exists():
        with zipfile.ZipFile(PIT / "c3d.zip") as z:
            z.extract(f"c3d/{fname[:6]}/{fname}", PIT)
    return ezc3d.c3d(str(path), extract_forceplat_data=True)


if __name__ == "__main__":
    DER.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    print("seed", SEED)
    poi = pd.read_csv(PIT / "poi_metrics.csv")
    meta = pd.read_csv(PIT / "metadata.csv")
    d = poi.merge(meta[["session_pitch", "user", "session_mass_kg", "session_height_m", "playing_level", "filename_new"]],
                  on="session_pitch")
    log = [f"{len(poi)} pitches in POI, {len(d)} after joining metadata"]
    n0 = len(d)
    d = d.dropna(subset=["lead_grf_mag_max"])
    log.append(f"dropped {n0 - len(d)} pitches with no force-plate data -> {len(d)} pitches")
    for leg in ["lead", "rear"]:
        d[f"{leg}_grf_bw"] = d[f"{leg}_grf_mag_max"] / (d.session_mass_kg * G)

    # ---- 1. cohort check, one row per pitcher (pitches within a pitcher are not independent)
    pp = d.groupby("user").agg(velo=("pitch_speed_mph", "mean"), mass=("session_mass_kg", "mean"),
                               lead_bw=("lead_grf_bw", "mean"), rear_bw=("rear_grf_bw", "mean"),
                               lead_n=("lead_grf_mag_max", "mean"), hand=("p_throws", "first"))
    log.append(f"{len(pp)} pitchers ({(pp.hand == 'R').sum()} R, {(pp.hand == 'L').sum()} L)")
    rows = []
    for lab, col in [("Lead-leg peak GRF (x bodyweight)", "lead_bw"), ("Lead-leg peak GRF (N)", "lead_n"),
                     ("Rear-leg peak GRF (x bodyweight)", "rear_bw"), ("Body mass (kg)", "mass")]:
        x, y = pp[col].values, pp.velo.values
        r = stats.pearsonr(x, y)
        lo, hi = boot_r(x, y, rng)
        rows.append([lab, r.statistic, f"{lo:.2f} to {hi:.2f}", f"{r.pvalue:.1e}", stats.spearmanr(x, y).statistic])
    cohort = pd.DataFrame(rows, columns=["Metric (per-pitcher mean)", "Pearson r", "95% bootstrap CI", "p", "Spearman"])
    res = lambda a, b: a - np.polyval(np.polyfit(b, a, 1), b)
    partial = np.corrcoef(res(pp.lead_n, pp.mass), res(pp.velo, pp.mass))[0, 1]

    # ---- 2. illustrative pair
    r_ = d[d.p_throws == "R"].copy()
    r_["dmed"] = (r_.pitch_speed_mph - r_.groupby("user").pitch_speed_mph.transform("median")).abs()
    rep = r_.sort_values(["user", "dmed", "session_pitch"]).groupby("user").head(1)
    pr = rep.merge(rep, on="playing_level", suffixes=("_h", "_l"))
    pr = pr[(pr.pitch_speed_mph_h > pr.pitch_speed_mph_l)
            & ((pr.session_mass_kg_h - pr.session_mass_kg_l).abs() <= MASS_TOL_KG)
            & ((pr.session_height_m_h - pr.session_height_m_l).abs() <= HEIGHT_TOL_M)]
    best = pr.assign(dv=pr.pitch_speed_mph_h - pr.pitch_speed_mph_l).sort_values("dv", ascending=False).iloc[0]
    pair = {"H": best.session_pitch_h, "L": best.session_pitch_l}
    log.append(f"{len(rep)} right-handed pitchers; {len(pr)} same-level pairs within {MASS_TOL_KG} kg and "
               f"{HEIGHT_TOL_M * 100:.0f} cm; picked largest velo gap")

    # ---- 3. signals for the pair
    fp = read_zip("force_plate")
    fp = fp[fp.session_pitch.isin(pair.values())]
    lm = read_zip("landmarks")
    lm = lm[lm.session_pitch.isin(pair.values())]
    jv = read_zip("joint_velos", usecols=["session_pitch", "time", "pelvis_velo_z", "torso_velo_z",
                                          "elbow_velo_x", "shoulder_velo_z", "BR_time"])
    jv = jv[jv.session_pitch.isin(pair.values())]
    # mean, not median, step: timestamps are rounded to 0.1 ms, which biases the median
    fp_hz, lm_hz = [1 / x.groupby("session_pitch").time.diff().mean() for x in (fp, lm)]
    log.append(f"sampling: force plate {fp_hz:.0f} Hz, joint positions {lm_hz:.0f} Hz; used as OBP provides them "
               f"(OBP pre-filtered: 4th-order Butterworth, 40 Hz force / 20 Hz joints); no further filtering here")

    grf, frames, velos, info, markers, plates_out = [], [], [], [], [], []
    for tag, sp in pair.items():
        p = d.set_index("session_pitch").loc[sp]
        m = p.session_mass_kg
        f = fp[fp.session_pitch == sp].copy()
        br = f.BR_time.iloc[0]
        # OBP stores lead force_x as + when braking; in the lab frame (+x toward home) the ground pushes the
        # planted foot back toward 2B, so the lab-frame reaction force is (-Fx, +Fy, +Fz). Checked two ways:
        # centre-of-mass x-acceleration, and the raw C3D plate forces below.
        f["t_ms"] = (f.time - br) * 1000
        f["gx_bw"], f["gy_bw"], f["gz_bw"] = -f.lead_force_x / (m * G), f.lead_force_y / (m * G), f.lead_force_z / (m * G)
        f["mag_bw"] = np.sqrt(f.lead_force_x ** 2 + f.lead_force_y ** 2 + f.lead_force_z ** 2) / (m * G)
        w = (f.time >= f.fp_10_time.iloc[0]) & (f.time <= f.MIR_time.iloc[0])   # lead foot contact -> max IR
        pk = f[w].mag_bw.idxmax()
        assert abs(f.loc[pk, "mag_bw"] * m * G - p.lead_grf_mag_max) < 1, "peak does not match OBP POI"
        t_pk = f.loc[pk, "time"]
        grf.append(f[["t_ms", "gx_bw", "gz_bw", "mag_bw"]].assign(tag=tag))
        s = lm[lm.session_pitch == sp].reset_index(drop=True)
        i_fr = int((s.time - t_pk).abs().argmin())                   # nearest 360 Hz frame (<1.4 ms away)
        frames.append(s.iloc[[i_fr]].drop(columns=["session_pitch"]).assign(tag=tag))
        v = jv[jv.session_pitch == sp].copy()
        v["t_ms"] = (v.time - v.BR_time) * 1000
        velos.append(v.drop(columns=["session_pitch", "time", "BR_time"]).assign(tag=tag))

        # raw C3D: markers at the drawn frame, plate geometry, COP. C3D frame i == landmarks row i.
        c = read_c3d(p.filename_new)
        lab = c["parameters"]["POINT"]["LABELS"]["value"]
        P = c["data"]["points"][:3]
        ank = (P[:, lab.index("LANK"), :] + P[:, lab.index("LMANK"), :]) / 2    # lead = left for a right-hander
        jc = s[["lead_ankle_jc_x", "lead_ankle_jc_y", "lead_ankle_jc_z"]].values.T
        align_mm = np.nanmedian(np.linalg.norm(ank - jc, axis=0)) * 1000
        assert P.shape[2] == len(s) and align_mm < 5, f"C3D/landmark frames misaligned ({align_mm:.1f} mm)"
        markers.append(pd.DataFrame(P[:, :, i_fr].T, columns=["x", "y", "z"]).assign(label=lab, tag=tag))
        pls = c["data"]["platform"]
        lead_pl = [j for j, pl in enumerate(pls) if pl["corners"][0].mean() > LANDING_X_M]
        k = int(round(t_pk * fp_hz))
        raw = sum(pls[j]["force"][:, k] for j in lead_pl)
        raw_diff = abs(np.linalg.norm(raw) - f.loc[pk, "mag_bw"] * m * G) / (f.loc[pk, "mag_bw"] * m * G)
        assert raw_diff < 0.02 and raw[0] < 0 < raw[2], f"raw C3D force disagrees with processed ({raw_diff:.1%})"
        fz = np.array([abs(pls[j]["force"][2, k]) for j in lead_pl])
        cop = sum(pls[j]["center_of_pressure"][:, k] * w_ for j, w_ in zip(lead_pl, fz)) / fz.sum()   # Fz-weighted
        for j, pl in enumerate(pls):
            plates_out.append(pd.DataFrame(pl["corners"].T, columns=["x", "y", "z"]).assign(
                plate=j, loaded=j in lead_pl and abs(pl["force"][2, k]) > 20, tag=tag))
        log.append(f"{tag} ({sp}): C3D markers vs model ankle {align_mm:.1f} mm; raw plate sum vs processed peak "
                   f"{raw_diff:.1%}; landing plates {lead_pl}")
        ang = np.degrees(np.arctan2(f.loc[pk, "gz_bw"], -f.loc[pk, "gx_bw"]))
        info.append([tag, sp, p.playing_level, p.pitch_speed_mph, m, p.session_height_m, f.loc[pk, "mag_bw"],
                     f.loc[pk, "gx_bw"], f.loc[pk, "gy_bw"], f.loc[pk, "gz_bw"], ang, p.lead_grf_angle_at_max,
                     (t_pk - br) * 1000, (s.time[i_fr] - t_pk) * 1000, *cop,
                     pp.loc[p.user, "velo"], pp.loc[p.user, "lead_bw"]])
    info = pd.DataFrame(info, columns=["tag", "session_pitch", "level", "velo", "mass_kg", "height_m", "peak_bw", "gx_bw",
                                       "gy_bw", "gz_bw", "angle_deg", "poi_angle_deg", "peak_ms_vs_BR", "frame_offset_ms",
                                       "cop_x", "cop_y", "cop_z", "pitcher_mean_velo", "pitcher_mean_lead_bw"])
    pd.concat(grf).to_csv(DER / "pitch_pair_grf.csv", index=False)
    pd.concat(frames).to_csv(DER / "pitch_pair_frames.csv", index=False)
    pd.concat(velos).to_csv(DER / "pitch_pair_velos.csv", index=False)
    pd.concat(markers).to_csv(DER / "pitch_pair_markers.csv", index=False)
    pd.concat(plates_out).to_csv(DER / "pitch_pair_plates.csv", index=False)
    info.to_csv(DER / "pitch_pair_info.csv", index=False)
    pp.assign(pair=pp.index.map({d.set_index("session_pitch").loc[sp, "user"]: t for t, sp in pair.items()})) \
      .to_csv(DER / "pitchers_cohort.csv")
    stats_out = {"r": cohort.iloc[0, 1], "ci": cohort.iloc[0, 2], "n": len(pp)}
    pd.Series(stats_out).to_csv(DER / "pitchers_cohort_stats.csv")

    show = info[["tag", "session_pitch", "level", "velo", "mass_kg", "height_m", "peak_bw", "angle_deg", "poi_angle_deg",
                 "peak_ms_vs_BR"]]
    txt = f"""# 04 Lead-leg block (OBP pitching biomechanics, a separate cohort)

{chr(10).join('- ' + s for s in log)}

## Does peak lead-leg force track velo? (n = {len(pp)} pitchers, seed {SEED})
{md_table(cohort)}

Partial r, lead-leg GRF (N) vs velo controlling for body mass: {partial:.2f}

## Illustrative pair (selected on level, handedness, mass, height and velo only)
{md_table(show, 2)}

`angle_deg` = force angle above horizontal in the side (x-z) view, recomputed here; `poi_angle_deg` = OBP's value.
`peak_ms_vs_BR` = time of peak lead-leg force relative to ball release (negative = before).
"""
    (RES / "04_lead_leg_block.md").write_text(txt, encoding="utf-8")
    print(txt)
