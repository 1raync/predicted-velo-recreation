"""Step 5b: does the s05 HIGH/LOW elbow-stress split hold up under a hierarchical Bayesian model (NumPyro)?
s05 averages each pitcher's 2-5 pitches and fits OLS on the 100 pitcher means. Here all 407 pitches are modelled:
    varus_pitch = expected(velo, mass, height) + u_pitcher + noise,   u_pitcher ~ Normal(0, tau)
and the posterior mean of u_pitcher replaces the OLS residual as "elbow stress beyond expected". This weights pitchers
by pitch count and pulls noisy (few-pitch) pitchers toward expected. Same thirds rule as s05, then the s05 group
comparison (Cohen's d, permutation p, FDR) is rerun on the new groups with the same permutations.
Outputs results/05b_elbow_bayes.md.        python src/s05b_elbow_bayes.py [--synthetic]   (--synthetic adds ~20 min)

DECIDED: sensitivity check, not the primary model. It moves few pitchers and changes no conclusion, and OLS on means is
  simpler to explain. Rejected: replacing the s05 split, which would tie the README numbers to prior/likelihood choices.
DECIDED: predictors are pitcher means (velo, mass, height), so "expected" means the same as in s05. Rejected: per-pitch
  velo, which mixes in a within-pitcher question (does the same pitcher load more on harder throws).
DECIDED: y and predictors standardised; a0, b ~ N(0, 1), tau, sigma ~ HalfNormal(1): weakly informative (the fit
  matches REML, which has no priors). Gaussian noise is primary; Student-t noise (nu ~ Gamma(2, 0.1)) is the robust
  sensitivity, since it discounts single outlier pitches. Pitcher effects are Normal in both.
DECIDED: centred pitcher effects + dense mass matrix. Non-centred with a diagonal mass matrix gave R-hat 1.016 (slopes
  and pitcher effects are correlated a posteriori). NUTS, 4 chains x 2000 warmup + 2000 draws, float64 (JAX's default
  is float32)."""
import sys
import numpyro
numpyro.set_host_device_count(4)
numpyro.enable_x64()
import jax
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import false_discovery_control
from common import ROOT, RES, SEED, md_table
from s05_elbow_stress import PIT, N_PERM, read_zip

PRED = ["pitch_speed_mph", "session_mass_kg", "session_height_m"]
LIK = {"Gaussian": False, "Student-t": True}


def hier_model(Xs, pid, J, y=None, student=False):
    a0 = numpyro.sample("a0", dist.Normal(0, 1))
    b = numpyro.sample("b", dist.Normal(0, 1).expand([Xs.shape[1]]).to_event(1))
    tau = numpyro.sample("tau", dist.HalfNormal(1))
    sig = numpyro.sample("sig", dist.HalfNormal(1))
    u = numpyro.sample("u", dist.Normal(0, tau).expand([J]).to_event(1))
    mu = (a0 + Xs @ b + u)[pid]
    lik = dist.StudentT(numpyro.sample("nu", dist.Gamma(2, 0.1)), mu, sig) if student else dist.Normal(mu, sig)
    numpyro.sample("y", lik.to_event(1), obs=y)


def fit(y, pid, Xp, student, seed, warm=2000, draws=2000):
    """y: pitch-level varus (Nm); pid: pitcher index of each pitch; Xp: pitcher-level predictors (J x 3).
    Returns posterior draws back in original units (Nm; Nm per mph, kg, m) plus sampler diagnostics."""
    ym, ys, xm, xs = y.mean(), y.std(), Xp.mean(0), Xp.std(0)
    mcmc = MCMC(NUTS(hier_model, target_accept_prob=0.9, dense_mass=True), num_warmup=warm, num_samples=draws,
                num_chains=4, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed), (Xp - xm) / xs, pid, len(Xp), y=(y - ym) / ys, student=student,
             extra_fields=("diverging",))
    s = {k: np.asarray(v) for k, v in mcmc.get_samples().items()}
    summ = numpyro.diagnostics.summary(mcmc.get_samples(group_by_chain=True))
    beta = ys * s["b"] / xs
    b0 = ym + ys * (s["a0"] - (s["b"] * xm / xs).sum(1))
    return dict(b0=b0, beta=beta, u=ys * s["u"], tau=ys * s["tau"], sig=ys * s["sig"], nu=s.get("nu"),
                rhat=max(np.max(v["r_hat"]) for v in summ.values()), ess=min(np.min(v["n_eff"]) for v in summ.values()),
                ndiv=int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum()))


def thirds(r):
    """Same rule as s05: bottom/top k of the ranked scores, k = what quantile(1/3) selects."""
    k3, rk = (len(r) - 1) // 3 + 1, r.rank(method="first")
    return pd.Series(np.where(rk > len(r) - k3, "HIGH", np.where(rk <= k3, "LOW", "MID")), index=r.index)


def compare(a, grp, cols, perms):
    """s05 step 2 (Cohen's d, permutation p, BH FDR) on the given grouping."""
    H, L = a[grp == "HIGH"], a[grp == "LOW"]
    both, k, rows = pd.concat([H, L]), len(H), []
    for c in cols:
        x, xh, xl = both[c].values, H[c].dropna(), L[c].dropna()
        sp = np.sqrt(((len(xh) - 1) * xh.var() + (len(xl) - 1) * xl.var()) / (len(xh) + len(xl) - 2))
        obs = xh.mean() - xl.mean()
        null = np.nanmean(x[perms][:, :k], 1) - np.nanmean(x[perms][:, k:], 1)
        rows.append([c, obs / sp if sp > 0 else np.nan, (np.sum(np.abs(null) >= abs(obs) - 1e-12) + 1) / (N_PERM + 1)])
    t = pd.DataFrame(rows, columns=["metric", "d", "p"]).set_index("metric")
    t["q"] = false_discovery_control(t.p)
    return t


ci = lambda x: f"{np.mean(x):.2f} [{np.quantile(x, .025):.2f}, {np.quantile(x, .975):.2f}]"


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    print("seed", SEED, "| numpyro", numpyro.__version__, "| jax", jax.__version__)
    # ---- same pitches and pitcher table as s05
    poi, meta = pd.read_csv(PIT / "poi_metrics.csv"), pd.read_csv(PIT / "metadata.csv")
    ev = read_zip("landmarks", usecols=["session_pitch", "fp_100_time", "BR_time"]).groupby("session_pitch").first()
    gap = ev.BR_time - ev.fp_100_time
    d = poi.merge(meta[["session_pitch", "user", "session_mass_kg", "session_height_m", "age_yrs"]], on="session_pitch")
    n0 = len(d)
    d = d[~d.session_pitch.isin(gap[(gap <= 0) | (gap > 0.4)].index)]
    num = d.select_dtypes("number").columns.drop(["session", "user"])
    a = d.groupby("user")[num].mean()
    npitch = d.groupby("user").size()
    log = [f"{n0} pitches -> {len(d)} after s05's event-timing drop; {len(a)} pitchers with "
           f"{npitch.min()}-{npitch.max()} pitches each"]
    assert d.elbow_varus_moment.notna().all() and a[PRED].notna().all().all()

    X = np.column_stack([np.ones(len(a)), a[PRED]])
    ols_resid = a.elbow_varus_moment - X @ np.linalg.lstsq(X, a.elbow_varus_moment, rcond=None)[0]
    grps = {"OLS (s05)": thirds(ols_resid)}

    # ---- hierarchical fits
    pid = a.index.get_indexer(d.user)
    y, Xp = d.elbow_varus_moment.values.astype(float), a[PRED].values.astype(float)
    fits, fit_rows = {}, []
    for lab, student in LIK.items():
        f = fits[lab] = fit(y, pid, Xp, student, SEED)
        u = pd.Series(f["u"].mean(0), index=a.index)
        grps[lab] = thirds(u)
        f["p_high"] = (f["u"] >= np.quantile(f["u"], 2 / 3, axis=1, keepdims=True)).mean(0)
        f["p_low"] = (f["u"] <= np.quantile(f["u"], 1 / 3, axis=1, keepdims=True)).mean(0)
        fit_rows.append([lab, f"{f['rhat']:.4f}", f"{f['ess']:.0f}", f["ndiv"], ci(f["beta"][:, 0]), ci(f["beta"][:, 1]),
                         ci(f["beta"][:, 2]), ci(f["tau"]), ci(f["sig"]), ci(f["nu"]) if student else "",
                         f"{np.corrcoef(ols_resid, u)[0, 1]:.4f}"])
    fit_tab = pd.DataFrame(fit_rows, columns=["noise", "max R-hat", "min ESS", "divergences", "Nm per mph",
                                              "Nm per kg", "Nm per m", "tau (Nm)", "sigma (Nm)", "nu", "r with OLS resid"])

    # second method: frequentist random-intercept model (REML) should match the Gaussian fit
    pdf = d[["user", "elbow_varus_moment"]].join(a[PRED].add_suffix("_m"), on="user")
    mx = smf.mixedlm("elbow_varus_moment ~ pitch_speed_mph_m + session_mass_kg_m + session_height_m_m", pdf,
                     groups=pdf.user).fit(reml=True)
    blup = pd.Series({k: v.iloc[0] for k, v in mx.random_effects.items()}).reindex(a.index)
    g = fits["Gaussian"]
    log.append("REML random-intercept model vs Gaussian fit: slopes " + ", ".join(f"{v:.2f}" for v in mx.fe_params[1:])
               + " vs " + ", ".join(f"{v:.2f}" for v in g["beta"].mean(0)) + " Nm per mph, kg, m; tau "
               f"{np.sqrt(mx.cov_re.iloc[0, 0]):.2f} vs {g['tau'].mean():.2f} Nm; pitcher effects r = "
               f"{np.corrcoef(blup, g['u'].mean(0))[0, 1]:.4f}, max |diff| {np.abs(blup - g['u'].mean(0)).max():.2f} Nm")

    # ---- group agreement with s05, and s05's group comparison rerun on each grouping
    agree = []
    for lab in LIK:
        f, gb, go = fits[lab], grps[lab], grps["OLS (s05)"]
        agree.append([lab, int((gb == go).sum()), ", ".join(f"{u} ({go[u]}->{gb[u]}, {npitch[u]} pitches)"
                                                         for u in a.index[gb != go]),
                      f"{(f['p_high'][gb == 'HIGH'] >= .8).sum()} of {(gb == 'HIGH').sum()}",
                      f"{(f['p_low'][gb == 'LOW'] >= .8).sum()} of {(gb == 'LOW').sum()}"])
    agree = pd.DataFrame(agree, columns=["noise", "same group as s05 (of 100)", "switched (s05 -> Bayes)",
                                         "HIGH with P(top third) >= 0.8", "LOW with P(bottom third) >= 0.8"])
    cols = [c for c in num if c not in PRED + ["age_yrs", "elbow_varus_moment"]]
    n_hl = int((grps["OLS (s05)"] != "MID").sum())
    assert all(int((g_ != "MID").sum()) == n_hl for g_ in grps.values())
    perms = np.array([rng.permutation(n_hl) for _ in range(N_PERM)])      # same permutations as s05
    tests = {lab: compare(a, gr, cols, perms) for lab, gr in grps.items()}
    sig = {lab: set(t.index[t.q < 0.05]) for lab, t in tests.items()}
    core = sorted(sig["OLS (s05)"], key=lambda m: -abs(tests["OLS (s05)"].d[m]))
    held = pd.concat({lab: t.loc[core, ["d", "q"]] for lab, t in tests.items()}, axis=1)
    held.columns = [f"{l} {c}" for l, c in held.columns]
    for lab in LIK:
        log.append(f"{lab}: {len(sig[lab] & sig['OLS (s05)'])} of the {len(core)} s05 metrics with q < 0.05 still have "
                   f"q < 0.05; newly significant: {sorted(sig[lab] - sig['OLS (s05)']) or 'none'}")

    # ---- optional: simulate from each fitted model on the real design (same X, same pitch counts), refit, check recovery
    synth = []
    if "--synthetic" in sys.argv:
        n_rep, sim_seed = 100, SEED + 1
        srng = np.random.default_rng(sim_seed)
        tert = lambda v: np.digitize(v, np.quantile(v, [1 / 3, 2 / 3]))
        for lab, student in LIK.items():
            f = fits[lab]
            tb0, tb, ttau, tsig = f["b0"].mean(), f["beta"].mean(0), f["tau"].mean(), f["sig"].mean()
            cover, rs, ag = [], [], []
            for r in range(n_rep):
                ut = srng.normal(0, ttau, len(a))
                noise = tsig * (srng.standard_t(f["nu"].mean(), len(y)) if student else srng.normal(0, 1, len(y)))
                gf = fit((tb0 + Xp @ tb + ut)[pid] + noise, pid, Xp, student, sim_seed + r, warm=1000, draws=1000)
                inside = lambda x, t: np.quantile(x, .025) <= t <= np.quantile(x, .975)
                cover.append([inside(gf["beta"][:, j], tb[j]) for j in range(3)] + [inside(gf["tau"], ttau), inside(gf["sig"], tsig)])
                rs.append(np.corrcoef(ut, gf["u"].mean(0))[0, 1])
                ag.append(np.mean(tert(ut) == tert(gf["u"].mean(0))))
            cv = np.array(cover).sum(0)
            synth.append(f"{lab}: 95% intervals covered the truth for mph {cv[0]}, kg {cv[1]}, m {cv[2]}, tau {cv[3]}, "
                         f"sigma {cv[4]} of {n_rep}; r(true, estimated pitcher effect) mean {np.mean(rs):.3f} "
                         f"(min {np.min(rs):.3f}); same third as the true effect for {np.mean(ag):.0%} of pitchers on average "
                         f"(worst {np.min(ag):.0%})")

    txt = f"""# 05b Elbow stress beyond expected: hierarchical Bayesian check of the s05 split (NumPyro)

Model (all pitches, not pitcher means): varus = expected(velo, mass, height) + u_pitcher + noise, u_pitcher ~ Normal(0, tau).
Score = posterior-mean u_pitcher; thirds by rank, as in s05. NUTS, 4 chains x 2000 draws, seed {SEED}.

{chr(10).join('- ' + s for s in log)}

## Fits (posterior mean [95% interval])
{md_table(fit_tab)}

## Group agreement with s05
{md_table(agree)}

## The s05 metrics with FDR q < 0.05, under each grouping (d = Cohen's d; q from {N_PERM} permutations + BH FDR)
{md_table(held.reset_index(names="metric"), 3)}

Synthetic recovery check (simulate from each fitted model, refit, compare with the known truth): run
`python src/s05b_elbow_bayes.py --synthetic` (~20+ min), which writes results/05b_synthetic.md (kept separate so
run_all doesn't overwrite it).
"""
    (RES / "05b_elbow_bayes.md").write_text(txt, encoding="utf-8")
    print(txt)
    if synth:
        s_txt = (f"# 05b Synthetic recovery check (seed {SEED + 1}; {n_rep} simulated datasets per model)\n\n"
                 "Each dataset reuses the real design (same velo/mass/height, same pitches per pitcher) with parameters set to "
                 "the fitted posterior means, then is refit with 1000 warmup + 1000 draws per chain.\n\n"
                 + "\n".join("- " + s for s in synth) + "\n")
        (RES / "05b_synthetic.md").write_text(s_txt, encoding="utf-8")
        print(s_txt)
