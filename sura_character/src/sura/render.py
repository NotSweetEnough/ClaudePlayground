"""Beauty / turntable renders of the finished character (Cycles)."""
import math
import os

import bpy

from . import butil, preview


def _clear_lights_cams():
    for o in list(bpy.data.objects):
        if o.type in ("LIGHT", "CAMERA"):
            bpy.data.objects.remove(o)


def stage(dark=True):
    _clear_lights_cams()
    w = bpy.data.worlds.new("Stage")
    w.use_nodes = True
    bg = w.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (0.018, 0.017, 0.02, 1) if dark else (0.2, 0.2, 0.22, 1)
    bg.inputs[1].default_value = 1.0
    bpy.context.scene.world = w
    # floor
    if "Floor" not in bpy.data.objects:
        bpy.ops.mesh.primitive_plane_add(size=12, location=(0, 0, 0))
        fl = bpy.context.active_object
        fl.name = "Floor"
        m = bpy.data.materials.new("M_Floor")
        p = m.node_tree.nodes["Principled BSDF"]
        p.inputs["Base Color"].default_value = (0.012, 0.011, 0.011, 1)
        p.inputs["Roughness"].default_value = 0.85
        fl.data.materials.append(m)
    butil.add_area_light("Key", (2.2, -2.6, 3.0), (0, 0, 1.2), 420, 1.6, (1.0, 0.93, 0.85))
    butil.add_area_light("Fill", (-3.0, -2.2, 1.6), (0, 0, 1.1), 70, 3.0, (0.75, 0.82, 1.0))
    butil.add_area_light("Rim", (-1.2, 2.8, 2.6), (0, 0, 1.5), 650, 1.2, (0.85, 0.9, 1.0))
    butil.add_area_light("RimR", (1.8, 2.4, 1.8), (0.3, 0, 1.2), 260, 1.0, (1.0, 0.45, 0.3))


def _orbit(name, target, dist, ang, lens, elev=0.0):
    a = math.radians(ang)
    loc = (target[0] + dist * math.sin(a), target[1] - dist * math.cos(a), target[2] + elev)
    return butil.add_camera(loc, target, lens, name)


def shot(path, target, dist, ang, lens=50, res=(1100, 1650), samples=96, elev=0.0):
    butil.setup_render(res, samples)
    _orbit("cam", target, dist, ang, lens, elev)
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    return path


def set_action(name):
    arm = bpy.data.objects["Sura_Rig"]
    arm.animation_data_create()
    arm.animation_data.action = bpy.data.actions.get(name) if name else None
    bpy.context.scene.frame_set(1)


def beauty(out_dir, samples=96):
    os.makedirs(out_dir, exist_ok=True)
    for o in bpy.data.objects:
        if o.type == "ARMATURE":
            o.hide_render = True
    stage()
    files = []
    tt = [shot(os.path.join(out_dir, f"_tt{a}.png"), (0, 0, 0.98), 5.2, a, 50, (700, 1300), samples, 0.25)
          for a in (0, 35, 90, 180, 300)]
    files.append(preview.montage(os.path.join(out_dir, "turntable.png"), tt))
    files.append(shot(os.path.join(out_dir, "hero_3q.png"), (0, 0, 1.0), 4.6, 28, 50, (1100, 1650), samples, 0.2))
    files.append(shot(os.path.join(out_dir, "head.png"), (0.0, -0.05, 1.83), 1.05, 30, 85, (1100, 1100), samples, 0.02))
    files.append(shot(os.path.join(out_dir, "head_back.png"), (0.0, 0.0, 1.6), 1.9, 160, 70, (1000, 1300), samples, 0.1))
    hand = bpy.data.objects["Sura_Rig"].data.bones["hand_l"].head_local
    files.append(shot(os.path.join(out_dir, "demon_arm.png"), (hand[0], hand[1], hand[2] - 0.02), 0.85, 55, 70,
                      (1100, 1100), samples, 0.05))
    set_action("Stance")
    files.append(shot(os.path.join(out_dir, "stance.png"), (0, 0, 1.0), 4.4, 20, 50, (1100, 1650), samples, 0.15))
    set_action(None)
    return files
