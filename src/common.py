"""Shared data loading, cleaning and CV setup. Every script imports from here so
all reported numbers come from the same cohort and the same folds."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "hp_obp.csv"
FIG = ROOT / "figures"
RES = ROOT / "results"
SEED = 0
TARGET = "pitch_speed_mph"

# Driveline's four "weighted heaviest" inputs (Predicted Pitch Velocity blog, 2021)
RAW = {
    "peak_power_[w]_mean_sj": "SJ_PP",                    # Squat Jump peak power (W)
    "rsi-modified_[m/s]_mean_cmj": "CMJ_RSImod",          # Countermovement Jump RSI-modified
    "best_rsi_(flight/contact_time)_mean_ht": "Hop_RSI",  # 10/5 Hop test RSI (ratio)
    "net_peak_vertical_force_[n]_max_imtp": "IMTP_NPF",   # Isometric mid-thigh pull net peak force (N)
}
BASE = list(RAW.values())
LABELS = {"SJ_PP": "SJ peak power", "CMJ_RSImod": "CMJ RSI-mod",
          "Hop_RSI": "Hop RSI", "IMTP_NPF": "IMTP net peak force"}
FOOT = ("Data: The OpenBiomechanics Project (Driveline Baseball R&D), CC BY-NC-SA 4.0. "
        "Independent recreation; not affiliated with or endorsed by Driveline Baseball.")


def load(one_per_athlete=True):
    """Load hp_obp.csv and apply the cleaning rules used everywhere in this repo.
    Returns (clean dataframe, number of raw rows)."""
    if not CSV.exists():
        raise FileNotFoundError("Run `python src/fetch_data.py` first.")
    raw = pd.read_csv(CSV)
    df = raw.rename(columns=RAW)
    df = df.dropna(subset=BASE + [TARGET, "body_weight_[lbs]"])
    df = df[df[TARGET] >= 60]            # 0 mph placeholders + a few implausible values
    df = df[df["CMJ_RSImod"] < 2]         # one physically implausible RSI-mod value
    if one_per_athlete:                   # retested athletes -> keep first test only
        df = df.sort_values("test_date").groupby("athlete_uid").head(1)
    df = df.reset_index(drop=True)
    df["BW_lb"] = df["body_weight_[lbs]"]
    df["BW_kg"] = df["BW_lb"] / 2.2046
    return add_bw_terms(df, bw_mean=df["BW_kg"].mean()), len(raw)


def add_bw_terms(df, bw_mean):
    d = df.copy()
    d["BW_c"] = d["BW_kg"] - bw_mean      # centred bodyweight (kg)
    d["BW_c2"] = d["BW_c"] ** 2           # curve term
    d["SJ_PP_perkg"] = d["SJ_PP"] / d["BW_kg"]
    d["IMTP_perkg"] = d["IMTP_NPF"] / d["BW_kg"]
    return d


FEATS_J = BASE + ["BW_c", "BW_c2"]       # final model: Driveline 4 + bodyweight curve


def folds(df):
    """10 x repeated 5-fold CV (one row per athlete, so folds never share an athlete)."""
    return list(RepeatedKFold(n_splits=5, n_repeats=10, random_state=SEED).split(df))


def md_table(df, floatfmt=3):
    """Tiny markdown table writer (avoids the optional `tabulate` dependency)."""
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda v: f"{v:.{floatfmt}f}" if pd.notna(v) else "")
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(map(str, r)) + " |" for r in df.astype(str).values]
    return "\n".join(lines)
