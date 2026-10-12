"""Preview / turntable rendering helpers (Cycles, CPU)."""
import math
import os

import bpy

from . import butil


def clay_material():
    mat = bpy.data.materials.new("Clay")
    if mat.node_tree is None:
        mat.use_nodes = True
    p = mat.node_tree.nodes["Principled BSDF"]
    p.inputs["Base Color"].default_value = (0.6, 0.55, 0.5, 1)
    p.inputs["Roughness"].default_value = 0.6
    return mat


def assign_clay(force=False):
    mat = clay_material()
    for o in bpy.data.objects:
        if o.type == "MESH" and (force or not o.data.materials):
            o.data.materials.clear()
            o.data.materials.append(mat)


def studio_lights(center=(0, 0, 1.0), scale=1.0, world=(0.2, 0.2, 0.22), world_strength=0.6):
    butil.studio_world(world, world_strength)
    cx, cy, cz = center
    butil.add_area_light("Key", (cx + 2 * scale, cy - 3 * scale, cz + 2 * scale), center, 600 * scale ** 2, 2 * scale)
    butil.add_area_light("Fill", (cx - 3 * scale, cy - 2 * scale, cz + 0.5 * scale), center, 200 * scale ** 2, 3 * scale)
    butil.add_area_light("Rim", (cx, cy + 3 * scale, cz + 2 * scale), center, 400 * scale ** 2, 2 * scale)


def render_orbit(path_prefix, target, dist, angles, lens=50, res=(800, 1200), samples=24, elev=0.0):
    """Render one image per yaw angle (0 = front, character faces -Y). Returns file paths."""
    butil.setup_render(res, samples)
    out = []
    for ang in angles:
        a = math.radians(ang)
        loc = (target[0] + dist * math.sin(a), target[1] - dist * math.cos(a), target[2] + elev)
        butil.add_camera(loc, target, lens, f"cam_{ang}")
        path = f"{path_prefix}_{ang:03d}.png"
        bpy.context.scene.render.filepath = path
        bpy.ops.render.render(write_still=True)
        out.append(path)
    return out


def montage(out, paths):
    from PIL import Image
    ims = [Image.open(p).convert("RGB") for p in paths]
    h = max(i.height for i in ims)
    c = Image.new("RGB", (sum(i.width for i in ims), h))
    x = 0
    for i in ims:
        c.paste(i, (x, 0))
        x += i.width
    c.save(out)
    for p in paths:
        if p != out:
            os.remove(p)
    return out
