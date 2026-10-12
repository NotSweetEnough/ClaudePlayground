"""Small Blender helpers shared by the build steps."""
import math

import bpy
import numpy as np
from mathutils import Vector


def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scn = bpy.context.scene
    scn.unit_settings.system = "METRIC"
    scn.unit_settings.scale_length = 1.0
    return scn


def mesh_object(name, verts, faces, uvs=None, collection=None, smooth=True):
    """verts (N,3); faces list of index lists; uvs: per-face list of (u,v) lists or None."""
    me = bpy.data.meshes.new(name)
    me.from_pydata([tuple(v) for v in np.asarray(verts, dtype=float)], [], [list(f) for f in faces])
    if uvs is not None:
        uvl = me.uv_layers.new(name="UVMap")
        flat = [uv for f in uvs for uv in f]
        uvl.data.foreach_set("uv", np.asarray(flat, dtype=np.float32).ravel())
    me.validate(clean_customdata=False)
    if smooth:
        me.shade_smooth()
    ob = bpy.data.objects.new(name, me)
    (collection or bpy.context.scene.collection).objects.link(ob)
    return ob


def get_verts(ob):
    n = len(ob.data.vertices)
    co = np.empty(n * 3, dtype=np.float64)
    ob.data.vertices.foreach_get("co", co)
    return co.reshape(n, 3)


def set_verts(ob, co):
    ob.data.vertices.foreach_set("co", np.asarray(co, dtype=np.float64).ravel())
    ob.data.update()


def get_normals(ob):
    me = ob.data
    n = len(me.vertices)
    no = np.empty(n * 3, dtype=np.float64)
    me.vertices.foreach_get("normal", no)
    return no.reshape(n, 3)


def set_active(ob, select_only=True):
    if select_only:
        for o in bpy.context.view_layer.objects:
            o.select_set(False)
    ob.select_set(True)
    bpy.context.view_layer.objects.active = ob


def apply_modifiers(ob):
    set_active(ob)
    for m in list(ob.modifiers):
        bpy.ops.object.modifier_apply(modifier=m.name)


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def look_at(ob, target):
    d = Vector(target) - ob.location
    ob.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()


def setup_render(res=(1080, 1620), samples=64, engine="CYCLES", transparent=False):
    scn = bpy.context.scene
    scn.render.engine = engine
    scn.render.resolution_x, scn.render.resolution_y = res
    scn.render.resolution_percentage = 100
    scn.render.film_transparent = transparent
    if engine == "CYCLES":
        scn.cycles.device = "CPU"
        scn.cycles.samples = samples
        scn.cycles.use_denoising = True
        try:
            scn.cycles.denoiser = "OPENIMAGEDENOISE"
        except TypeError:
            pass
        scn.cycles.max_bounces = 6
        scn.cycles.transparent_max_bounces = 32
    scn.view_settings.view_transform = "AgX"
    scn.view_settings.look = "AgX - Medium High Contrast"
    return scn


def add_camera(loc, target, lens=85, name="Camera"):
    cam = bpy.data.cameras.new(name)
    cam.lens = lens
    ob = bpy.data.objects.new(name, cam)
    bpy.context.scene.collection.objects.link(ob)
    ob.location = loc
    look_at(ob, target)
    bpy.context.scene.camera = ob
    return ob


def add_area_light(name, loc, target, energy, size=1.0, color=(1, 1, 1)):
    li = bpy.data.lights.new(name, "AREA")
    li.energy = energy
    li.size = size
    li.color = color
    ob = bpy.data.objects.new(name, li)
    bpy.context.scene.collection.objects.link(ob)
    ob.location = loc
    look_at(ob, target)
    return ob


def studio_world(color=(0.05, 0.05, 0.06), strength=1.0):
    w = bpy.data.worlds.new("World")
    w.use_nodes = True
    bg = w.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (*color, 1)
    bg.inputs[1].default_value = strength
    bpy.context.scene.world = w
    return w


def deg(x):
    return math.radians(x)
