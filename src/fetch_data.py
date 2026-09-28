"""Download the OpenBiomechanics High Performance data, pinned to a fixed commit
so results are reproducible. The data is NOT committed to this repo."""
import urllib.request
from common import ROOT

COMMIT = "44b98dae05cceb016f080ab39d105c85b8639084"   # OBP main, 2026-09-07
BASE_URL = (f"https://raw.githubusercontent.com/drivelineresearch/openbiomechanics/"
            f"{COMMIT}/high_performance/data/")
FILES = ["hp_obp.csv", "data_dictionary.csv"]

if __name__ == "__main__":
    out = ROOT / "data"
    out.mkdir(exist_ok=True)
    for f in FILES:
        urllib.request.urlretrieve(BASE_URL + f, out / f)
        print("downloaded", out / f)
    print("Data license: CC BY-NC-SA 4.0 + professional-organization exclusion. See "
          "https://github.com/drivelineresearch/openbiomechanics/blob/main/LICENSE-DATA.md")
