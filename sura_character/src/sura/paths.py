import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MH_DIR = os.environ.get("SURA_MH_DIR", os.path.join(ROOT, "third_party", "makehuman"))
BUILD_DIR = os.environ.get("SURA_BUILD_DIR", os.path.join(ROOT, "build"))
EXPORT_DIR = os.path.join(ROOT, "export")
TEX_DIR = os.path.join(EXPORT_DIR, "textures")
RENDER_DIR = os.path.join(ROOT, "renders")
