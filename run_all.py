"""Reproduce every result and figure:  python run_all.py"""
import runpy, sys
from pathlib import Path

SRC = Path(__file__).parent / "src"
sys.path.insert(0, str(SRC))
STEPS = ["fetch_data", "s01_baseline_model", "s02_f_tests", "s03_bodyweight", "fig_two_athletes", "fig_model"]
for s in STEPS:
    if s == "fetch_data" and (Path(__file__).parent / "data" / "hp_obp.csv").exists():
        print("data already downloaded, skipping fetch"); continue
    print(f"\n===== {s} =====")
    runpy.run_path(str(SRC / f"{s}.py"), run_name="__main__")
