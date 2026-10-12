"""Build the Sura character end to end.

    pip install -r requirements.txt          # bpy (Blender as a Python module), numpy, scipy, pillow
    python build.py                          # full build: 2K textures, GLB + FBX + LODs + .blend + renders
    python build.py --tex 4096               # 4K texture sets
    python build.py --quick                  # 512px textures, no LODs / renders (fast iteration)
"""
import argparse
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

import bpy  # noqa: E402

from sura import assemble, export, finalize, paths, render  # noqa: E402
from sura.assemble import log  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", type=int, default=2048, help="skin / outfit texture size")
    ap.add_argument("--hair-tex", type=int, default=2048)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-renders", action="store_true")
    ap.add_argument("--out", default=paths.EXPORT_DIR)
    args = ap.parse_args()
    if args.quick:
        args.tex = args.hair_tex = 512

    out = args.out
    tex_dir = os.path.join(out, "textures")
    if os.path.isdir(tex_dir):
        shutil.rmtree(tex_dir)
    os.makedirs(tex_dir, exist_ok=True)

    parts = assemble.build()
    F = finalize.finalize(parts, tex_dir, args.tex, args.hair_tex)
    arm = F["arm"]
    meshes = [F["body"], F["outfit"], F["hair"], F["eyes"], F["mouth"]]

    log("actions")
    export.make_actions(arm)

    log(f"LOD0 triangles: {export.tri_count(meshes):,}")
    for o in meshes:
        log(f"   {o.name:14s} {export.tri_count([o]):>8,} tris  "
            f"{len(o.data.shape_keys.key_blocks) - 1 if o.data.shape_keys else 0:>3} blendshapes  "
            f"materials: {', '.join(m.name for m in o.data.materials)}")
    log(f"bones: {len(arm.data.bones)}")

    log("export: GLB")
    export.export_glb(os.path.join(out, "Sura.glb"), [arm] + meshes)
    log("export: FBX")
    export.export_fbx(os.path.join(out, "Sura.fbx"), [arm] + meshes)

    if not args.quick:
        for ratio, sfx in ((0.5, "_LOD1"), (0.2, "_LOD2")):
            lod = export.make_lod(meshes, arm, ratio, sfx)
            log(f"export: {sfx[1:]} ({export.tri_count(lod):,} tris)")
            export.export_fbx(os.path.join(out, f"Sura{sfx}.fbx"), [arm] + lod, animations=False)
            for o in lod:
                bpy.data.objects.remove(o)

    log("save .blend")
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(out, "Sura.blend"), relative_remap=True, compress=True)

    log("export: web GLB (1K JPEG textures)")
    web = os.path.join(out, "web")
    os.makedirs(web, exist_ok=True)
    for img in bpy.data.images:
        if img.size[0] > 1024:
            img.scale(1024, 1024)
    export.export_glb(os.path.join(web, "Sura_web.glb"), [arm] + meshes, image_format="JPEG", jpeg_quality=88)

    if not (args.quick or args.no_renders):
        bpy.ops.wm.open_mainfile(filepath=os.path.join(out, "Sura.blend"))
        render.beauty(paths.RENDER_DIR)
    log("done")


if __name__ == "__main__":
    main()
