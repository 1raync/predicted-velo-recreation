"""Step 5: elbow stress beyond expected, in OBP's pitching biomechanics dataset (a different set of athletes
from the force-plate tests).
  1. Per pitcher, regress peak elbow varus moment on velo, mass and height; the residual is "elbow stress beyond
     expected". Split into top vs bottom thirds (HIGH vs LOW), which leaves the groups matched on velo/mass/height.
  2. Compare the groups on every OBP point-of-interest metric: Cohen's d, permutation p, Benjamini-Hochberg FDR.
  3. Layback (max shoulder external rotation) vs varus moment, adjusted for velo, mass and height; then the whole
     layback curve, HIGH vs LOW, with an spm1d two-sample t-test (random-field-theory correction over time).
  4. Matched right-handed HIGH/LOW pairs; draw the pair with the largest residual gap at max layback.
Outputs results/05_elbow_stress.md and data/derived/elbow_pair_*.csv

DECIDED: drop the 4 pitches whose event timing is broken (foot plant after release, or > 0.4 s before it).
DECIDED: one row per pitcher (pitch means). Within-pitcher variation is small next to between-pitcher variation.
DECIDED: residual on velo, mass and height, not raw varus. Raw varus tracks body size and velo, so a raw split
  would compare bigger, faster pitchers with smaller, slower ones.
DECIDED: pair = largest residual gap among right-handed HIGH/LOW pairs matched within 1.5 mph, 5 kg and 5 cm at the
  same level. Each pitcher is drawn with the pitch closest to their own median varus moment, frozen at max layback.
DECIDED: layback curve test on -300..+90 ms around release, in real ms (not time-normalised), so it lines up with
  events; +90 ms is the latest point every recording covers (spm1d needs complete curves). Each pitch is linearly
  interpolated onto one 360 Hz grid aligned to release (a sub-frame shift, not a change of rate), then averaged per
  pitcher. spm1d default unequal-variance t-test, two-tailed, alpha 0.05; equal-variance reported as a sensitivity."""
import zipfile
import ezc3d
import numpy as np
import pandas as pd
import spm1d
from scipy import stats
from scipy.stats import false_discovery_control
from common import ROOT, RES, SEED, md_table

PIT = ROOT / "data" / "pitching"
DER = ROOT / "data" / "derived"
N_PERM = 5000
MATCH = {"pitch_speed_mph": 1.5, "session_mass_kg": 5, "session_height_m": 0.05}
LANDING_X_M = 1.0
CURVE_MS = (-300, 90)
SHOW = {"max_shoulder_external_rotation": "Max layback (deg)", "stride_angle": "Stride angle (deg, + = cross-body)",
        "torso_rotation_br": "Torso rotation at release (deg)", "max_elbow_extension_velo": "Max elbow extension velo (deg/s)",
        "shoulder_absorption_fp_br": "Shoulder energy absorbed, foot plant to release (J)",
        "shoulder_internal_rotation_moment": "Shoulder internal rotation moment (Nm)",
        "timing_peak_torso_to_peak_pelvis_rot_velo": "Pelvis-to-torso peak timing (s)",
        "max_rotation_hip_shoulder_separation": "Hip-shoulder separation (deg)"}


def read_zip(name, **kw):
    z = zipfile.ZipFile(PIT / f"{name}.zip")
    return pd.read_csv(z.open(z.namelist()[0]), **kw)


def read_c3d(fname):
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
    lm = read_zip("landmarks")
    ev = lm.groupby("session_pitch")[["fp_100_time", "BR_time", "MER_time"]].first()
    gap = ev.BR_time - ev.fp_100_time
    bad = sorted(gap[(gap <= 0) | (gap > 0.4)].index)
    d = poi.merge(meta[["session_pitch", "user", "session_mass_kg", "session_height_m", "age_yrs", "playing_level",
                        "filename_new"]], on="session_pitch")
    n0 = len(d)
    d = d[~d.session_pitch.isin(bad)]
    log = [f"{n0} pitches; dropped {n0 - len(d)} with broken event timing {bad} -> {len(d)}"]

    # ---- 1. per-pitcher residual and tertile split
    num = d.select_dtypes("number").columns.drop(["session", "user"])
    a = d.groupby("user")[num].mean()
    a["hand"], a["level"] = d.groupby("user").p_throws.first(), d.groupby("user").playing_level.first()
    X = np.column_stack([np.ones(len(a)), a.pitch_speed_mph, a.session_mass_kg, a.session_height_m])
    beta, *_ = np.linalg.lstsq(X, a.elbow_varus_moment, rcond=None)
    a["expected"] = X @ beta
    a["resid"] = a.elbow_varus_moment - a.expected
    r2 = 1 - a.resid.var() / a.elbow_varus_moment.var()
    lo, hi = a.resid.quantile([1 / 3, 2 / 3])
    a["grp"] = np.where(a.resid >= hi, "HIGH", np.where(a.resid <= lo, "LOW", "MID"))
    H, L = a[a.grp == "HIGH"], a[a.grp == "LOW"]
    log.append(f"{len(a)} pitchers; expected varus = {beta[0]:.1f} + {beta[1]:.2f}*mph + {beta[2]:.2f}*kg "
               f"{beta[3]:+.1f}*m (R2 {r2:.2f}, residual SD {a.resid.std():.1f} Nm)")
    log.append(f"HIGH = residual >= {hi:+.1f} Nm (n={len(H)}), LOW = residual <= {lo:+.1f} Nm (n={len(L)}); middle third dropped")

    # ---- 2. group comparison on every POI metric (same permutations reused for every metric)
    both = pd.concat([H, L])
    k = len(H)
    perms = np.array([rng.permutation(len(both)) for _ in range(N_PERM)])
    rows = []
    for c in [c for c in num if c not in ("session_pitch",)]:
        x = both[c].values
        ok = ~np.isnan(x)
        xh, xl = H[c].dropna(), L[c].dropna()
        sp = np.sqrt(((len(xh) - 1) * xh.var() + (len(xl) - 1) * xl.var()) / (len(xh) + len(xl) - 2))
        obs = xh.mean() - xl.mean()
        xp = x[perms]
        null = np.nanmean(xp[:, :k], 1) - np.nanmean(xp[:, k:], 1)
        p = (np.sum(np.abs(null) >= abs(obs) - 1e-12) + 1) / (N_PERM + 1)
        rows.append([c, xh.mean(), xl.mean(), obs / sp if sp > 0 else np.nan, p, stats.spearmanr(a[c], a.resid, nan_policy="omit").statistic])
    comp = pd.DataFrame(rows, columns=["metric", "HIGH", "LOW", "d", "p_perm", "rho_all"])
    design = ["pitch_speed_mph", "session_mass_kg", "session_height_m", "age_yrs", "elbow_varus_moment", "expected", "resid"]
    tested = comp[~comp.metric.isin(design)].copy()
    tested["q"] = false_discovery_control(tested.p_perm)
    tested = tested.reindex(tested.d.abs().sort_values(ascending=False).index)
    log.append(f"{len(tested)} metrics compared; {int((tested.q < 0.05).sum())} with FDR q < 0.05")
    balance = comp[comp.metric.isin(design[:5] + ["resid"])]

    # ---- 3. layback vs varus, adjusted
    res = lambda y, Z: y - np.column_stack([np.ones(len(y)), Z]) @ np.linalg.lstsq(np.column_stack([np.ones(len(y)), Z]), y, rcond=None)[0]
    Z = a[["pitch_speed_mph", "session_mass_kg", "session_height_m"]].values
    pr_ = stats.pearsonr(res(a.max_shoulder_external_rotation.values, Z), res(a.elbow_varus_moment.values, Z))

    # whole layback curve (shoulder_angle_z = external rotation; its per-pitch max equals OBP's max layback exactly)
    ja = read_zip("joint_angles", usecols=["session_pitch", "time", "BR_time", "shoulder_angle_z"])
    ja = ja[ja.session_pitch.isin(d.session_pitch)]
    assert (ja.groupby("session_pitch").shoulder_angle_z.max()
            - d.set_index("session_pitch").max_shoulder_external_rotation).abs().max() < 1e-6
    step = 1000 / 360
    ms = np.arange(CURVE_MS[0], CURVE_MS[1] + 1e-6, step)
    curves = pd.DataFrame({sp: np.interp(ms / 1000, g.time - g.BR_time, g.shoulder_angle_z, left=np.nan, right=np.nan)
                           for sp, g in ja.groupby("session_pitch")}).T
    C = curves.join(d.set_index("session_pitch").user).groupby("user").mean()
    YH, YL = C.loc[H.index].values, C.loc[L.index].values
    assert not (np.isnan(YH).any() or np.isnan(YL).any()), "incomplete curves in window"
    spm = {}
    for lab, eq in [("unequal variance (primary)", False), ("equal variance", True)]:
        ti = spm1d.stats.ttest2(YH, YL, equal_var=eq).inference(alpha=0.05, two_tailed=True)
        spm[lab] = (ti.zstar, [(ms[0] + c.endpoints[0] * step, ms[0] + c.endpoints[1] * step, c.P,
                                "HIGH lower" if c.csign < 0 else "HIGH higher") for c in ti.clusters])
    i_mer = np.argmax((YH.mean(0) + YL.mean(0)) / 2)
    log.append(f"layback curves: {CURVE_MS[0]}..+{CURVE_MS[1]} ms around release, linear interpolation onto a common "
               f"360 Hz grid aligned to release, per-pitcher means (HIGH {len(YH)}, LOW {len(YL)})")

    # ---- 4. matched right-handed pairs
    R = a[a.hand == "R"].reset_index()
    pairs = R[R.grp == "HIGH"].merge(R[R.grp == "LOW"], on="level", suffixes=("_h", "_l"))
    for c, tol in MATCH.items():
        pairs = pairs[(pairs[f"{c}_h"] - pairs[f"{c}_l"]).abs() <= tol]
    pairs["dres"] = pairs.resid_h - pairs.resid_l
    less_lb = (pairs.max_shoulder_external_rotation_h < pairs.max_shoulder_external_rotation_l)
    log.append(f"{len(pairs)} matched right-handed HIGH/LOW pairs; HIGH has less layback in {less_lb.sum()} of {len(pairs)}")
    best = pairs.sort_values("dres", ascending=False).iloc[0]
    users = {"H": best.user_h, "L": best.user_l}

    frames, markers, plates_out, info = [], [], [], []
    for tag, u in users.items():
        dp = d[d.user == u]
        pitch = dp.iloc[(dp.elbow_varus_moment - dp.elbow_varus_moment.median()).abs().argsort().iloc[0]]
        sp = pitch.session_pitch
        s = lm[lm.session_pitch == sp].reset_index(drop=True)
        i_fr = int((s.time - s.MER_time.iloc[0]).abs().argmin())
        frames.append(s.iloc[[i_fr]].drop(columns=["session_pitch"]).assign(tag=tag))
        c = read_c3d(pitch.filename_new)
        lab = c["parameters"]["POINT"]["LABELS"]["value"]
        P = c["data"]["points"][:3]
        el = (P[:, lab.index("RELB"), :] + P[:, lab.index("RMELB"), :]) / 2     # throwing elbow of a right-hander
        align_mm = np.nanmedian(np.linalg.norm(el - s[["elbow_jc_x", "elbow_jc_y", "elbow_jc_z"]].values.T, axis=0)) * 1000
        assert P.shape[2] == len(s) and align_mm < 5, f"C3D/landmark frames misaligned ({align_mm:.1f} mm)"
        markers.append(pd.DataFrame(P[:, :, i_fr].T, columns=["x", "y", "z"]).assign(label=lab, tag=tag))
        k_fp = int(round(s.time[i_fr] * 1080))
        for j, pl in enumerate(c["data"]["platform"]):
            plates_out.append(pd.DataFrame(pl["corners"].T, columns=["x", "y", "z"]).assign(
                plate=j, loaded=pl["corners"][0].mean() > LANDING_X_M and abs(pl["force"][2, k_fp]) > 20, tag=tag))
        r = a.loc[u]
        info.append([tag, u, sp, r.level, r.pitch_speed_mph, r.session_mass_kg, r.session_height_m, r.elbow_varus_moment,
                     r.expected, r.resid, r.max_shoulder_external_rotation, pitch.elbow_varus_moment,
                     pitch.max_shoulder_external_rotation, (s.time[i_fr] - s.BR_time.iloc[0]) * 1000, align_mm])
        log.append(f"{tag} (pitcher {u}, pitch {sp}): throwing-elbow markers vs model elbow {align_mm:.1f} mm")
    info = pd.DataFrame(info, columns=["tag", "user", "session_pitch", "level", "velo", "mass_kg", "height_m", "varus_nm",
                                       "expected_nm", "resid_nm", "layback_deg", "pitch_varus_nm", "pitch_layback_deg",
                                       "frame_ms_vs_BR", "align_mm"])
    pd.concat(frames).to_csv(DER / "elbow_pair_frames.csv", index=False)
    pd.concat(markers).to_csv(DER / "elbow_pair_markers.csv", index=False)
    pd.concat(plates_out).to_csv(DER / "elbow_pair_plates.csv", index=False)
    info.to_csv(DER / "elbow_pair_info.csv", index=False)

    key = tested[tested.metric.isin(SHOW)].assign(metric=lambda t: t.metric.map(SHOW))
    txt = f"""# 05 Elbow stress beyond expected (OBP pitching biomechanics, a separate cohort)

{chr(10).join('- ' + s for s in log)}

## Group balance (HIGH vs LOW, per-pitcher means)
{md_table(balance.drop(columns="rho_all"), 2)}

## Selected metrics (d = Cohen's d; p from {N_PERM} permutations, seed {SEED}; q = FDR across all {len(tested)} metrics)
{md_table(key, 3)}

Layback vs peak varus moment, adjusted for velo, mass and height: partial r = {pr_.statistic:.2f} (p = {pr_.pvalue:.1e}).

## Whole layback curve, HIGH vs LOW (spm1d two-sample t-test, two-tailed, alpha 0.05)
{chr(10).join(f"- {k}: critical t = {z:.2f}; significant clusters: " + ("; ".join(f"{s:.0f} to {e:.0f} ms ({sg}, p = {p:.1e})" for s, e, p, sg in cl) or "none") for k, (z, cl) in spm.items())}
- group-mean curves peak at {ms[i_mer]:.0f} ms: HIGH {YH.mean(0)[i_mer]:.1f} deg vs LOW {YL.mean(0)[i_mer]:.1f} deg

## All metrics with q < 0.15
{md_table(tested[tested.q < 0.15], 3)}

## Drawn pair (largest residual gap among the matched pairs; pitcher means, plus the drawn pitch)
{md_table(info.drop(columns=["align_mm"]), 1)}
"""
    (RES / "05_elbow_stress.md").write_text(txt, encoding="utf-8")
    print(txt)
