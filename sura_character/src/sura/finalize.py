"""UVs, texture bakes, final materials and object merge into the export layout."""
import math
import os

import bpy
import numpy as np

from . import geo, materials, texgen
from .butil import get_verts, set_active, smoothstep
from .paths import MH_DIR


def _edit_all(ob):
    set_active(ob)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")


def pack_uvs(ob, margin=0.003):
    _edit_all(ob)
    bpy.ops.uv.select_all(action="SELECT")
    bpy.ops.uv.average_islands_scale()
    bpy.ops.uv.pack_islands(rotate=True, margin=margin)
    bpy.ops.object.mode_set(mode="OBJECT")


def smart_uvs(ob, angle=60.0, margin=0.002):
    _edit_all(ob)
    bpy.ops.uv.smart_project(angle_limit=math.radians(angle), island_margin=margin, area_weight=1.0,
                             correct_aspect=True, scale_to_bounds=False)
    bpy.ops.uv.select_all(action="SELECT")
    bpy.ops.uv.pack_islands(rotate=True, margin=margin)
    bpy.ops.object.mode_set(mode="OBJECT")


def hem_field(ob, rings=5):
    faces = [tuple(p.vertices) for p in ob.data.polygons]
    n = len(ob.data.vertices)
    h = geo.boundary_mask(faces, n).astype(float)
    A = geo.adjacency(faces, n)
    deg = np.maximum(np.asarray(A.sum(1)).ravel(), 1)
    for _ in range(rings):
        h = np.maximum(h, 0.75 * (A @ h) / deg)
    return h


def tag_garment(ob, kind, grime_scale=1.0):
    co = get_verts(ob)
    gid = materials.GARMENT_IDS[kind] / 10.0
    grime = smoothstep(0.58, 0.02, co[:, 2]) * 0.9 * grime_scale
    hem = hem_field(ob) if len(ob.data.polygons) > 50 else np.zeros(len(co))
    materials.set_color_attr(ob, "garm", np.stack([np.full(len(co), gid), grime, hem, np.ones(len(co))], 1))


def recalc_normals(ob):
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bmesh.ops.triangulate(bm, faces=[f for f in bm.faces if len(f.verts) > 4])
    bm.to_mesh(ob.data)
    bm.free()


def recalc_ngons(ob):
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    bmesh.ops.triangulate(bm, faces=[f for f in bm.faces if len(f.verts) > 4])
    bm.to_mesh(ob.data)
    bm.free()


def assign(ob, mat):
    ob.data.materials.clear()
    ob.data.materials.append(mat)


def finalize(P, tex_dir, size=2048, hair_size=2048):
    from .assemble import join, log
    arm = P["arm"]
    bpy.data.objects.remove(P["body_full"])
    B = P["body"]

    # --- skin ----------------------------------------------------------------------
    log("skin: masks + UV pack")
    materials.body_masks(B, arm)
    pack_uvs(B)
    hidden = [P["hair_cap"], P["hair_cards"], P["hair_fly"], P["lashes"], P["brows"]]
    log(f"skin: baking {size}px")
    skin = materials.bake_set([B], materials.SkinLook, "T_Sura_Skin", size, tex_dir, ao=True, emission=True,
                              bump_distance=0.0012, hide=hidden)

    # --- outfit --------------------------------------------------------------------
    log("outfit: tagging + join + UVs")
    for o in (P["tunic"], P["skirt"]):
        tag_garment(o, "cotton", 0.3)
    tag_garment(P["pants"], "pants")
    tag_garment(P["wraps"], "wraps")
    for o in [P["belt"]] + P["knot"] + P["tails"]:
        tag_garment(o, "sash", 0.3)
    for o in P["straps"]:
        tag_garment(o, "straps")
    for o in P["shoes"]:
        tag_garment(o, "canvas" if o.name.startswith("Shoe_") else "rubber" if o.name.startswith("Sole_") else "laces")
    for o in P["shoes"] + P["knot"]:
        recalc_normals(o)
    garments = [P["tunic"], P["skirt"], P["pants"], P["wraps"], P["belt"]] + P["knot"] + P["tails"] + P["straps"] + P["shoes"]
    outfit = join(garments, "Sura_Outfit")
    recalc_ngons(outfit)
    smart_uvs(outfit)
    log(f"outfit: baking {size}px")
    out_tex = materials.bake_set([outfit], materials.OutfitLook, "T_Sura_Outfit", size, tex_dir, ao=True,
                                 bump_distance=0.0015, hide=hidden)

    # --- generated textures --------------------------------------------------------
    log("hair / horn / eye textures")
    hair_c, hair_n = texgen.hair_atlas(os.path.join(tex_dir, "T_Sura_Hair_BaseColor.png"),
                                       os.path.join(tex_dir, "T_Sura_Hair_Normal.png"), hair_size)
    horn = texgen.horn_maps(os.path.join(tex_dir, "T_Sura_Horn_BaseColor.png"),
                            os.path.join(tex_dir, "T_Sura_Horn_Normal.png"),
                            os.path.join(tex_dir, "T_Sura_Horn_ORM.png"))
    eye = texgen.eye_texture(os.path.join(MH_DIR, "eyes", "materials", "brown_eye.png"),
                             os.path.join(tex_dir, "T_Sura_Eye_BaseColor.png"))

    # --- final materials ---------------------------------------------------------------
    log("materials")
    M_skin = materials.pbr_material("M_Sura_Skin", skin["basecolor"], skin["normal"], skin["orm"],
                                    skin["emissive"], emissive_strength=3.0)
    M_out = materials.pbr_material("M_Sura_Outfit", out_tex["basecolor"], out_tex["normal"], out_tex["orm"],
                                   double_sided=True)
    M_hair = materials.pbr_material("M_Sura_Hair", hair_c, hair_n, roughness=0.36, alpha_clip=0.35,
                                    double_sided=True)
    M_horn = materials.pbr_material("M_Sura_Horn", *horn)
    M_eye = materials.pbr_material("M_Sura_Eye", eye, roughness=0.06)
    M_teeth = materials.pbr_material("M_Sura_Teeth", color=materials.LIN((226, 218, 196)), roughness=0.25)
    M_tongue = materials.pbr_material("M_Sura_Tongue", color=materials.LIN((150, 66, 66)), roughness=0.4)

    for a in ("maskA", "maskB", "maskC"):
        if a in B.data.color_attributes:
            B.data.color_attributes.remove(B.data.color_attributes[a])
    if "garm" in outfit.data.color_attributes:
        outfit.data.color_attributes.remove(outfit.data.color_attributes["garm"])

    assign(B, M_skin)
    assign(outfit, M_out)
    for o in (P["lashes"], P["brows"], P["hair_cap"], P["hair_cards"], P["hair_fly"]):
        assign(o, M_hair)
    horns = join(P["claws"] + P["spikes"], "Horns")
    recalc_normals(horns)
    assign(horns, M_horn)
    assign(P["eyes"], M_eye)
    M_cornea = materials.pbr_material("M_Sura_Cornea", color=(1, 1, 1), roughness=0.02, alpha=0.06)
    eyes_me = P["eyes"].data
    eyes_me.materials.append(M_cornea)
    uvl = eyes_me.uv_layers[0].data
    for poly in eyes_me.polygons:
        uv = np.mean([uvl[li].uv for li in poly.loop_indices], axis=0)
        if np.hypot(uv[0] - 0.933, uv[1] - 0.067) < 0.07:
            poly.material_index = 1
    assign(P["teeth"], M_teeth)
    assign(P["tongue"], M_tongue)

    log("joining export meshes")
    body = join([B, P["brows"], P["lashes"], horns], "Sura_Body")
    hair = join([P["hair_cards"], P["hair_cap"], P["hair_fly"]], "Sura_Hair")
    mouth = join([P["teeth"], P["tongue"]], "Sura_Mouth")
    eyes = P["eyes"]
    eyes.name = eyes.data.name = "Sura_Eyes"
    arm.name = "Sura_Rig"
    arm.data.name = "Sura_Rig"
    final = [body, outfit, hair, eyes, mouth]
    for ob in final:
        for m in list(ob.modifiers):
            ob.modifiers.remove(m)
        for l in list(ob.data.uv_layers)[1:]:
            ob.data.uv_layers.remove(l)
        ob.data.uv_layers[0].name = "UVMap"
        mod = ob.modifiers.new("Armature", "ARMATURE")
        mod.object = arm
        ob.parent = arm
        ob.data.shade_smooth()
    return dict(body=body, outfit=outfit, hair=hair, eyes=eyes, mouth=mouth, arm=arm)
