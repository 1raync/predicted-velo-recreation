"""Reproduce every result and figure:  python run_all.py"""
import runpy, sys
from pathlib import Path

SRC = Path(__file__).parent / "src"
sys.path.insert(0, str(SRC))
STEPS = ["fetch_data",   # skips files already downloaded
         "s01_baseline_model", "s02_f_tests", "s03_bodyweight", "s04_lead_leg_block",
         "fig_two_athletes", "fig_model", "fig_lead_leg", "fig_kinematic_sequence"]
for s in STEPS:
    print(f"\n===== {s} =====")
    runpy.run_path(str(SRC / f"{s}.py"), run_name="__main__")
