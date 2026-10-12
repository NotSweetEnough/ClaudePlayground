"""Vendor the subset of MakeHuman (CC0) data this build uses into third_party/makehuman.

    python tools/fetch_makehuman.py                 # sparse-clone the pinned commit from GitHub
    python tools/fetch_makehuman.py --from <clone>  # copy from an existing makehuman checkout
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

REPO = "https://github.com/makehumancommunity/makehuman"
COMMIT = "a8bc2d54ff0ac92e78ff71431b1023eda42bf482"
DEST = os.path.join(ROOT, "third_party", "makehuman")

STATIC = [
    "3dobjs/base.obj",
    "rigs/default.mhskel",
    "rigs/default_weights.mhw",
    "eyes/high-poly/high-poly.mhclo",
    "eyes/high-poly/high-poly.obj",
    "eyes/materials/brown_eye.png",
]


def needed(src_data):
    from sura import body  # noqa: imports bpy-free helpers only at call time
    files = list(STATIC)
    files += ["targets/" + rel for rel, w in body.all_targets() if abs(w) > 1e-6]
    units = os.path.join(src_data, "targets", "expression", "units")
    for race in ("african", "asian", "caucasian"):
        for f in sorted(os.listdir(os.path.join(units, race))):
            if f.endswith(".target"):
                files.append(f"targets/expression/units/{race}/{f}")
    return sorted(set(files))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", help="existing makehuman repo checkout")
    a = ap.parse_args()
    tmp = None
    if a.src:
        src = a.src
    else:
        tmp = tempfile.mkdtemp()
        src = os.path.join(tmp, "makehuman")
        subprocess.check_call(["git", "clone", "--filter=blob:none", "--sparse", "--no-checkout", REPO, src])
        subprocess.check_call(["git", "-C", src, "sparse-checkout", "set", "makehuman/data/3dobjs", "makehuman/data/rigs",
                               "makehuman/data/eyes", "makehuman/data/targets"])
        subprocess.check_call(["git", "-C", src, "checkout", COMMIT])
    data = os.path.join(src, "makehuman", "data")
    os.environ["SURA_MH_DIR"] = data
    files = needed(data)
    for rel in files:
        d = os.path.join(DEST, rel)
        os.makedirs(os.path.dirname(d), exist_ok=True)
        shutil.copy2(os.path.join(data, rel), d)
    print(f"copied {len(files)} files to {DEST}")
    if tmp:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    main()
