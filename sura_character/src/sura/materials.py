"""Procedural look-dev baked to game-ready PBR texture sets (Cycles, CPU).

Each texture set is authored as procedural shader channels driven by object-space
noise + per-vertex masks, then baked to: BaseColor (sRGB), Normal (tangent, OpenGL
+Y), ORM (R=AO, G=Roughness, B=Metallic) and Emissive (sRGB) where needed.
"""
import math
import os

import bpy
import numpy as np

from . import demon, hair
from .butil import get_verts, set_active, smoothstep
from .nodes import Tree

LIN = lambda c: tuple(((x / 255.0 + 0.055) / 1.055) ** 2.4 if x > 10 else x / 255.0 / 12.92 for x in c)


# --------------------------------------------------------------------------------------
# per-vertex masks

def set_color_attr(ob, name, rgba):
    me = ob.data
    if name in me.color_attributes:
        me.color_attributes.remove(me.color_attributes[name])
    a = me.color_attributes.new(name, "FLOAT_COLOR", "POINT")
    a.data.foreach_set("color", np.asarray(rgba, dtype=np.float32).ravel())


def _g(co, c, r):
    return np.exp(-np.sum((co - np.asarray(c)) ** 2, 1) / (2 * r * r))


def body_masks(B, arm):
    co = get_verts(B)
    x, y, z = co[:, 0], co[:, 1], co[:, 2]
    ax = np.abs(x)
    face = y < -0.10
    # A: demon / lips / eye sockets / redness
    dem = demon.demon_mask(arm, co)
    s, d, L_up = demon.arm_param(arm, co)
    tip = np.array(arm.data.bones["middle_03_l"].tail_local)
    L_tot = L_up + np.linalg.norm(np.array(arm.data.bones["lowerarm_l"].tail_local) - np.array(arm.data.bones["lowerarm_l"].head_local)) \
        + np.linalg.norm(tip - np.array(arm.data.bones["lowerarm_l"].tail_local))
    armT = np.clip((s - L_up) / (L_tot - L_up), 0, 1) * (x > 0.12)
    mz = 1.751
    hz = np.where(z > mz, 0.0095, 0.0115)
    lip_e = (x / 0.0225) ** 2 + ((z - mz) / hz) ** 2
    lips = smoothstep(1.25, 0.8, lip_e) * (y < -0.135)
    eye = np.zeros(len(co))
    for sx in (1, -1):
        c = np.array([0.032 * sx, -0.136, 1.83])
        dd = np.linalg.norm(co - c, axis=1)
        eye = np.maximum(eye, smoothstep(0.036, 0.016, dd) * face * (0.7 + 0.3 * (z < 1.83)))
    nose_cand = np.where((ax < 0.006) & (z > 1.77) & (z < 1.82), y, 1.0)
    nose = co[np.argmin(nose_cand)]
    red = 0.8 * _g(co, nose, 0.012)
    for sx in (1, -1):
        red += 0.55 * _g(co, (0.045 * sx, -0.15, 1.795), 0.016)
        red += 0.7 * ((ax > 0.068) & (z > 1.785) & (z < 1.875) & (y > -0.085) & (y < 0.01)) * smoothstep(0.068, 0.08, ax)
    knuckle = np.zeros(len(co))
    for side in ("l", "r"):
        for f in ("index", "middle", "ring", "pinky", "thumb"):
            for k in (1, 2, 3):
                b = arm.data.bones.get(f"{f}_{k:02d}_{side}")
                if b is not None:
                    knuckle = np.maximum(knuckle, _g(co, b.head_local, 0.008))
    red = np.clip(red + 0.5 * knuckle, 0, 1)
    # B: brows / stubble / scalp / vein zone
    t = (ax - 0.011) / 0.054
    zc = 1.8485 + 0.0065 * np.clip(t, 0, 1) + 0.0035 * np.sin(np.pi * np.clip(t, 0, 1))
    brow = smoothstep(0.0065, 0.0025, np.abs(z - zc)) * (t > -0.05) * (t < 1.05) * (y < -0.12) * (1 - 0.4 * np.clip(t, 0, 1))
    stub = smoothstep(1.695, 1.715, z) * smoothstep(1.785, 1.765, z) * (y < -0.06) * smoothstep(0.08, 0.06, ax)
    stub *= 1 - lips
    stub = np.clip(stub, 0, 1) * 0.7
    scalp = hair.scalp_mask(co)
    veinz = smoothstep(L_up - 0.22, L_up - 0.03, s) * (x > 0.1) * (1 - dem)
    veinz = np.maximum(veinz, smoothstep(0.11, 0.0, np.linalg.norm((co - [0.05, -0.08, 1.56]) * [1, 1, 0.7], axis=1)) * (x > -0.01))
    # C: arm param / forehead / nasolabial / knuckles
    fore = smoothstep(1.86, 1.875, z) * smoothstep(1.915, 1.9, z) * (y < -0.12) * smoothstep(0.06, 0.035, ax)
    naso = np.zeros(len(co))
    for sx in (1, -1):
        a = np.array([0.019 * sx, -0.168, 1.787])
        b = np.array([0.030 * sx, -0.156, 1.748])
        ab = b - a
        tt = np.clip(((co - a) @ ab) / (ab @ ab), 0, 1)
        dd = np.linalg.norm(co - (a + tt[:, None] * ab), axis=1)
        naso = np.maximum(naso, smoothstep(0.004, 0.0, dd) * face)
    set_color_attr(B, "maskA", np.stack([dem, lips, eye, red], 1))
    set_color_attr(B, "maskB", np.stack([brow, stub, scalp, veinz], 1))
    set_color_attr(B, "maskC", np.stack([armT, fore, naso, knuckle], 1))


# --------------------------------------------------------------------------------------
# skin look-dev (channels)

class SkinLook:
    def __init__(self, T):
        self.T = T
        P = T.obj_coords()
        self.P = P
        A, Bm, C = T.attr("maskA"), T.attr("maskB"), T.attr("maskC")
        self.demon, self.lips, self.eye = T.sep(A.outputs["Color"])
        self.red = A.outputs["Alpha"]
        self.brow, self.stub, self.scalp = T.sep(Bm.outputs["Color"])
        self.veinz = Bm.outputs["Alpha"]
        self.armT, self.fore, self.naso = T.sep(C.outputs["Color"])
        self.knuck = C.outputs["Alpha"]
        # shared patterns
        self.cells_edge = T.voronoi(P, 115.0, "DISTANCE_TO_EDGE")
        self.crack = T.inv(T.smooth(0.0, 0.07, self.cells_edge))
        self.big_cells = T.voronoi(P, 38.0, "DISTANCE_TO_EDGE")
        self.big_crack = T.inv(T.smooth(0.0, 0.05, self.big_cells))
        self.vein = T.ridged(P, 16.0, 3.0, 10.0, 0.4)
        self.vein_fine = T.ridged(P, 45.0, 2.0, 14.0, 0.3)
        self.glow_n = T.noise(P, 7.0, 2.0)

    def color(self):
        T = self.T
        P = self.P
        c = T.rgb(LIN((196, 156, 138)))
        mott = T.noise(P, 38.0, 4.0, 0.6)
        c = T.mix(T.remap(mott, 0.3, 0.7, 0.0, 1.0), T.mix(0.5, c, T.rgb(LIN((200, 165, 152)))), T.mix(0.5, c, T.rgb(LIN((222, 192, 178)))))
        blotch = T.smooth(0.55, 0.78, T.noise(P, 9.0, 3.0))
        c = T.mix(T.mul(blotch, 0.35), c, T.rgb(LIN((205, 140, 128))))
        c = T.mix(T.mul(self.red, 0.55), c, T.rgb(LIN((200, 120, 108))))
        c = T.mix(T.mul(self.eye, 0.75), c, T.mix(0.5, c, T.rgb(LIN((120, 92, 98)))), "MULTIPLY")
        c = T.mix(T.mul(self.lips, 0.85), c, T.rgb(LIN((150, 82, 78))))
        dots = T.smooth(0.25, 0.0, T.voronoi(P, 900.0, "F1"))
        c = T.mix(T.mul(self.stub, T.add(0.35, T.mul(dots, 0.65))), c, T.rgb(LIN((150, 146, 148))))
        strands = T.smooth(0.45, 0.75, T.noise(T.vmul(P, (1.0, 0.25, 6.0)), 160.0, 2.0))
        c = T.mix(T.mul(self.brow, T.add(0.35, T.mul(strands, 0.6))), c, T.rgb(LIN((88, 86, 88))))
        c = T.mix(T.mul(self.scalp, 0.8), c, T.rgb(LIN((128, 124, 124))))
        c = T.mix(T.mul(self.naso, 0.25), c, T.rgb(LIN((170, 128, 118))))
        # creeping corruption veins
        v = T.mul(T.add(self.vein, T.mul(self.vein_fine, 0.6)), self.veinz, clamp=True)
        c = T.mix(T.mul(v, 0.9), c, T.rgb(LIN((70, 8, 14))))
        # demon arm: crimson scaled hide, black cracks, darkening to the claws
        plates = T.mix(T.noise(P, 25.0, 3.0), T.rgb(LIN((92, 10, 9))), T.rgb(LIN((52, 5, 5))))
        plates = T.mix(T.mul(T.inv(T.smooth(0.0, 0.25, self.cells_edge)), 0.45), plates, T.rgb(LIN((122, 20, 15))))
        ash = T.smooth(0.6, 0.8, T.noise(P, 14.0, 4.0))
        plates = T.mix(T.mul(ash, 0.35), plates, T.rgb(LIN((60, 48, 46))))
        dark = T.smooth(0.45, 0.95, self.armT)
        plates = T.mix(T.mul(dark, 0.85), plates, T.rgb(LIN((26, 4, 4))))
        plates = T.mix(T.mul(self.knuck, 0.6), plates, T.rgb(LIN((30, 5, 5))))
        dc = T.mix(T.mul(T.add(self.crack, self.big_crack), 0.9, clamp=True), plates, T.rgb(LIN((14, 2, 2))))
        dc = T.mix(T.mul(self.vein, 0.7), dc, T.rgb(LIN((40, 3, 4))))
        return T.mix(self.demon, c, dc)

    def emission(self):
        T = self.T
        g = T.mul(T.add(T.mul(self.crack, 0.6), self.big_crack), T.smooth(0.55, 0.7, self.glow_n), clamp=True)
        g = T.mul(g, self.demon)
        g = T.mul(g, T.inv(T.mul(T.smooth(0.6, 1.0, self.armT), 0.7)))
        v = T.mul(T.mul(self.vein, self.veinz), T.smooth(0.55, 0.7, self.glow_n))
        g = T.add(g, T.mul(v, 0.25))
        return T.mix(g, T.rgb((0, 0, 0)), T.rgb(LIN((255, 96, 24))))

    def roughness(self):
        T = self.T
        r = T.add(0.50, T.mul(T.noise(self.P, 20.0, 2.0), 0.12))
        r = T.mixf(self.lips, r, 0.38)
        r = T.mixf(T.mul(self.red, 0.4), r, 0.44)
        dr = T.add(0.46, T.mul(self.crack, 0.35))
        return T.mixf(self.demon, r, dr)

    def height(self):
        T = self.T
        P = self.P
        pores = T.smooth(0.0, 0.4, T.voronoi(P, 650.0, "F1"))
        fine = T.add(T.noise(P, 260.0, 2.0), T.mul(T.ridged(P, 90.0, 2.0, 6.0), 0.35))
        skin = T.add(T.mul(pores, 0.35), T.mul(fine, 0.2))
        wr = T.m("SINE", T.add(T.mul(T.sepxyz(P)[2], 2 * math.pi / 0.0075), T.mul(T.noise(P, 30.0, 2.0), 4.0)))
        skin = T.add(skin, T.mul(T.mul(wr, self.fore), 0.25))
        skin = T.sub(skin, T.mul(self.naso, 0.5))
        bumps = T.noise(P, 60.0, 3.0)
        dem = T.add(T.mul(T.inv(T.add(self.crack, self.big_crack)), 0.9), T.mul(bumps, 0.35))
        dem = T.add(dem, T.mul(self.vein, 0.5))
        skin = T.add(skin, T.mul(T.mul(T.add(self.vein, self.vein_fine), self.veinz), 0.45))
        return T.mixf(self.demon, T.mul(skin, 0.8), dem)


# --------------------------------------------------------------------------------------
# outfit look-dev

GARMENT_IDS = {"cotton": 1, "pants": 2, "wraps": 3, "sash": 4, "straps": 5, "canvas": 6, "rubber": 7, "laces": 8}


class OutfitLook:
    def __init__(self, T, ao_img=None):
        self.T = T
        P = T.obj_coords()
        self.P = P
        g = T.attr("garm")
        gid, self.grime, self.hem = T.sep(g.outputs["Color"])
        gid8 = T.mul(gid, 10.0)
        self.is_ = {k: T.mul(T.m("COMPARE", gid8, float(v), 0.5), 1.0) for k, v in GARMENT_IDS.items()}
        self.ao = None
        if ao_img is not None:
            self.ao = T.image(ao_img, T.uv(), non_color=True).outputs["Color"]
        self.grain = T.noise(P, 240.0, 2.0)
        self.wrinkle = T.ridged(P, 22.0, 3.0, 4.0, 0.6)
        self.dirt_n = T.noise(P, 6.0, 4.0, 0.65)

    def _sel(self, base, key, val):
        return self.T.mix(self.is_[key], base, val)

    def color(self):
        T = self.T
        cot = T.rgb(LIN((198, 192, 178)))
        cot = T.mix(T.remap(T.noise(self.P, 12.0, 3.0), 0.3, 0.7, 0, 1), T.mix(0.5, cot, T.rgb(LIN((186, 180, 166)))), cot)
        c = cot
        c = self._sel(c, "pants", T.mix(0.25, cot, T.rgb(LIN((180, 176, 166)))))
        c = self._sel(c, "wraps", T.rgb(LIN((206, 202, 190))))
        blk = T.mix(T.mul(T.smooth(0.55, 0.8, self.dirt_n), 0.6), T.rgb(LIN((22, 21, 20))), T.rgb(LIN((44, 42, 40))))
        c = self._sel(c, "sash", blk)
        c = self._sel(c, "straps", T.rgb(LIN((18, 17, 16))))
        c = self._sel(c, "canvas", T.rgb(LIN((214, 212, 204))))
        c = self._sel(c, "rubber", T.rgb(LIN((196, 190, 176))))
        c = self._sel(c, "laces", T.rgb(LIN((222, 220, 214))))
        # dust and wear
        dust = T.mul(self.grime, T.remap(self.dirt_n, 0.3, 0.75, 0.25, 1.0), clamp=True)
        c = T.mix(T.mul(dust, 0.75), c, T.rgb(LIN((128, 112, 92))))
        cloth = T.add(T.add(self.is_["cotton"], self.is_["pants"]), self.is_["wraps"])
        c = T.mix(T.mul(T.mul(self.hem, cloth), 0.3), c, T.rgb(LIN((150, 140, 120))))
        c = T.mix(T.mul(T.mul(self.hem, self.is_["sash"]), 0.25), c, T.rgb(LIN((52, 50, 47))))
        if self.ao is not None:
            cloth = T.add(T.add(T.add(self.is_["cotton"], self.is_["pants"]), self.is_["wraps"]), self.is_["canvas"])
            c = T.mix(T.mul(T.mul(T.inv(T.remap(self.ao, 0.4, 1.0, 0, 1)), 0.35), cloth), c, T.rgb(LIN((120, 108, 90))))
            c = T.mix(1.0, c, T.mix(0.0, T.rgb((1, 1, 1)), self.ao), "MULTIPLY")
        return c

    def roughness(self):
        T = self.T
        r = T.add(0.84, T.mul(self.grain, 0.08))
        r = T.mixf(self.is_["sash"], r, 0.62)
        r = T.mixf(self.is_["straps"], r, 0.7)
        r = T.mixf(self.is_["rubber"], r, 0.58)
        return r

    def height(self):
        T = self.T
        P = self.P
        weave = T.mul(T.add(T.wave(T.uv(), 520.0, "BANDS", "X"), T.wave(T.uv(), 520.0, "BANDS", "Y")), 0.5)
        h = T.add(T.mul(self.grain, 0.25), T.mul(weave, 0.2))
        h = T.add(h, T.mul(self.wrinkle, 0.55))
        sash_creases = T.ridged(T.vmul(P, (2.0, 2.0, 0.25)), 40.0, 2.0, 5.0, 0.4)
        h = T.mixf(self.is_["sash"], h, T.add(T.mul(sash_creases, 0.6), T.mul(self.grain, 0.15)))
        twist = T.wave(T.uv(), 90.0, "BANDS", "DIAGONAL")
        h = T.mixf(self.is_["straps"], h, T.mul(twist, 0.6))
        ribs = T.wave(P, 180.0, "BANDS", "X")
        h = T.mixf(self.is_["rubber"], h, T.add(T.mul(ribs, 0.15), T.mul(self.grain, 0.1)))
        return h


# --------------------------------------------------------------------------------------
# baking

def new_image(name, size, non_color=False, alpha=False):
    if name in bpy.data.images:
        bpy.data.images.remove(bpy.data.images[name])
    img = bpy.data.images.new(name, size, size, alpha=alpha, float_buffer=False)
    if non_color:
        img.colorspace_settings.name = "Non-Color"
    return img


def bake(objs, build, img, kind="EMIT", samples=1, margin=12):
    """Bake a procedural channel. build(Tree) must leave an Output node wired."""
    mat = bpy.data.materials.new("_bake")
    T = Tree(mat)
    build(T)
    node = T.node("ShaderNodeTexImage", image=img)
    T.t.nodes.active = node
    saved = {}
    for o in objs:
        saved[o.name] = [s.material for s in o.material_slots]
        o.data.materials.clear()
        o.data.materials.append(mat)
    set_active(objs[0])
    for o in objs:
        o.select_set(True)
    scn = bpy.context.scene
    scn.render.engine = "CYCLES"
    scn.cycles.device = "CPU"
    scn.cycles.samples = samples
    scn.render.bake.margin = margin
    scn.render.bake.use_clear = True
    kw = {}
    if kind == "NORMAL":
        kw = dict(normal_space="TANGENT")
    bpy.ops.object.bake(type=kind, **kw)
    for o in objs:
        o.data.materials.clear()
        for m in saved[o.name]:
            o.data.materials.append(m)
    bpy.data.materials.remove(mat)
    return img


def save_png(img, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.filepath_raw = path
    img.file_format = "PNG"
    img.save()
    return path


def pixels(img):
    a = np.empty(img.size[0] * img.size[1] * 4, dtype=np.float32)
    img.pixels.foreach_get(a)
    return a.reshape(img.size[1], img.size[0], 4)


def pack_orm(ao, rough, size, path, name):
    img = new_image(name, size, non_color=True)
    out = np.ones((size, size, 4), dtype=np.float32)
    out[..., 0] = pixels(ao)[..., 0] if ao is not None else 1.0
    out[..., 1] = pixels(rough)[..., 0]
    out[..., 2] = 0.0
    img.pixels.foreach_set(out.ravel())
    save_png(img, path)
    return img


def bake_set(objs, look_cls, prefix, size, tex_dir, ao=True, emission=False, bump_distance=0.001,
             ao_samples=48, look_kwargs=None, hide=()):
    """Bake BaseColor / Normal / ORM (/ Emissive) for objects sharing one UV atlas."""
    look_kwargs = look_kwargs or {}
    for o in hide:
        o.hide_render = True
    res = {}
    ao_img = None
    if ao:
        ao_img = new_image(f"{prefix}_AO", size, non_color=True)

        def b_ao(T):
            p = T.node("ShaderNodeBsdfPrincipled")
            out = T.node("ShaderNodeOutputMaterial")
            T.link(p.outputs[0], out.inputs["Surface"])
        if bpy.context.scene.world is None:
            bpy.context.scene.world = bpy.data.worlds.new("World")
        bpy.context.scene.world.light_settings.distance = 0.25
        bake(objs, b_ao, ao_img, "AO", samples=ao_samples)
    col = new_image(f"{prefix}_BaseColor", size)
    bake(objs, lambda T: T.emission_out(look_cls(T, **({"ao_img": ao_img} if look_cls is OutfitLook else {})).color()), col)
    res["basecolor"] = save_png(col, os.path.join(tex_dir, f"{prefix}_BaseColor.png"))
    rough = new_image(f"{prefix}_Rough", size, non_color=True)
    bake(objs, lambda T: T.emission_out(look_cls(T).roughness()), rough)
    nrm = new_image(f"{prefix}_Normal", size, non_color=True)
    bake(objs, lambda T: T.bump_out(look_cls(T).height(), bump_distance), nrm, "NORMAL", samples=4)
    res["normal"] = save_png(nrm, os.path.join(tex_dir, f"{prefix}_Normal.png"))
    res["orm"] = os.path.join(tex_dir, f"{prefix}_ORM.png")
    pack_orm(ao_img, rough, size, res["orm"], f"{prefix}_ORM")
    if emission:
        em = new_image(f"{prefix}_Emissive", size)
        bake(objs, lambda T: T.emission_out(look_cls(T).emission()), em)
        res["emissive"] = save_png(em, os.path.join(tex_dir, f"{prefix}_Emissive.png"))
    for o in hide:
        o.hide_render = False
    return res


# --------------------------------------------------------------------------------------
# final (exportable) materials

def gltf_settings_group():
    name = "glTF Material Output"
    if name in bpy.data.node_groups:
        return bpy.data.node_groups[name]
    g = bpy.data.node_groups.new(name, "ShaderNodeTree")
    g.interface.new_socket("Occlusion", in_out="INPUT", socket_type="NodeSocketFloat")
    g.interface.new_socket("Thickness", in_out="INPUT", socket_type="NodeSocketFloat")
    return g


def load_image(path, non_color=False):
    img = bpy.data.images.load(path, check_existing=True)
    if non_color:
        img.colorspace_settings.name = "Non-Color"
    return img


def pbr_material(name, basecolor=None, normal=None, orm=None, emissive=None, emissive_strength=1.0,
                 color=(0.8, 0.8, 0.8), roughness=0.5, metallic=0.0, alpha_clip=None, double_sided=False,
                 normal_strength=1.0, alpha=None):
    mat = bpy.data.materials.new(name)
    T = Tree(mat)
    p = T.node("ShaderNodeBsdfPrincipled")
    out = T.node("ShaderNodeOutputMaterial")
    T.link(p.outputs[0], out.inputs["Surface"])
    p.inputs["Base Color"].default_value = (*color, 1)
    p.inputs["Roughness"].default_value = roughness
    p.inputs["Metallic"].default_value = metallic
    if basecolor:
        n = T.image(load_image(basecolor))
        T.link(n.outputs["Color"], p.inputs["Base Color"])
        if alpha_clip is not None:
            # glTF exporter recognises 1 - (alpha < cutoff) as alphaMode MASK
            lt = T.m("LESS_THAN", n.outputs["Alpha"], alpha_clip)
            T.link(T.m("SUBTRACT", 1.0, lt), p.inputs["Alpha"])
    if normal:
        n = T.image(load_image(normal, True), non_color=True)
        nm = T.node("ShaderNodeNormalMap")
        nm.inputs["Strength"].default_value = normal_strength
        T.link(n.outputs["Color"], nm.inputs["Color"])
        T.link(nm.outputs["Normal"], p.inputs["Normal"])
    if orm:
        n = T.image(load_image(orm, True), non_color=True)
        sc = T.node("ShaderNodeSeparateColor")
        T.link(n.outputs["Color"], sc.inputs[0])
        T.link(sc.outputs[1], p.inputs["Roughness"])
        T.link(sc.outputs[2], p.inputs["Metallic"])
        g = T.node("ShaderNodeGroup")
        g.node_tree = gltf_settings_group()
        T.link(sc.outputs[0], g.inputs["Occlusion"])
    if emissive:
        n = T.image(load_image(emissive))
        T.link(n.outputs["Color"], p.inputs["Emission Color"])
        p.inputs["Emission Strength"].default_value = emissive_strength
    if alpha is not None:
        p.inputs["Alpha"].default_value = alpha
        mat.surface_render_method = "BLENDED"
    if alpha_clip is not None:
        mat.surface_render_method = "DITHERED"
    mat.use_backface_culling = not double_sided
    return mat
