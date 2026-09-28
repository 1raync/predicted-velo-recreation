"""Download the OpenBiomechanics data, pinned to a fixed commit / release so results
are reproducible. The data is NOT committed to this repo. Files already present are skipped."""
import hashlib
import urllib.request
from common import ROOT

COMMIT = "44b98dae05cceb016f080ab39d105c85b8639084"   # OBP main, 2026-09-07
RAW_URL = f"https://raw.githubusercontent.com/drivelineresearch/openbiomechanics/{COMMIT}/"
REL_URL = "https://github.com/drivelineresearch/openbiomechanics/releases/download/dataset-v1/"
FILES = {  # local path under data/ -> source URL
    "hp_obp.csv": RAW_URL + "high_performance/data/hp_obp.csv",
    "data_dictionary.csv": RAW_URL + "high_performance/data/data_dictionary.csv",
    "pitching/metadata.csv": RAW_URL + "baseball_pitching/data/metadata.csv",
    "pitching/poi_metrics.csv": RAW_URL + "baseball_pitching/data/poi/poi_metrics.csv",
    # full-signal tables and raw C3D files live on the dataset-v1 release (~274 MB zipped)
    "pitching/force_plate.zip": REL_URL + "pitching_force_plate.zip",
    "pitching/landmarks.zip": REL_URL + "pitching_landmarks.zip",
    "pitching/joint_velos.zip": REL_URL + "pitching_joint_velos.zip",
    "pitching/c3d.zip": REL_URL + "pitching_c3d.zip",
}
# from scripts/release_checksums.sha256 at the pinned commit
SHA256 = {
    "pitching/c3d.zip": "2f5277dc63c7f7535fa4b15310d6f38b3f3b761e572896eec529854c8c920d29",
    "pitching/force_plate.zip": "f36a1722e23101be2387acc8f91ec38d2ee5d640388f688af7fd8fa52eef29ed",
    "pitching/landmarks.zip": "efcc078d5d8e77ed1fbf0b9f6a78f750c57a624165f71f30e34142b74db7ec22",
    "pitching/joint_velos.zip": "5752f571b40acc4f02afc42dcfdb4b9eb191fcffd5fb25bc3c585e149941fb69",
}

if __name__ == "__main__":
    out = ROOT / "data"
    for f, url in FILES.items():
        dest = out / f
        if dest.exists():
            print("already have", dest); continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, dest)
        if f in SHA256 and hashlib.sha256(dest.read_bytes()).hexdigest() != SHA256[f]:
            dest.unlink()
            raise RuntimeError(f"checksum mismatch for {f}; deleted it, re-run to retry")
        print("downloaded", dest)
    print("Data license: CC BY-NC-SA 4.0 + professional-organization exclusion. See "
          "https://github.com/drivelineresearch/openbiomechanics/blob/main/LICENSE-DATA.md")
