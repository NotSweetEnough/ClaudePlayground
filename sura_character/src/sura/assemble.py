"""Builds the complete rigged character (geometry + skeleton + skin weights + blendshapes)."""
import os
import time

import bpy
import bmesh
import numpy as np
from scipy.spatial import cKDTree

from . import body, butil, clothing, demon, geo, hair, mh, rig
from .butil import get_verts, set_active, set_verts, smoothstep


def log(msg, t0=[time.time()]):
    print(f"[sura {time.time() - t0[0]:7.1f}s] {msg}", flush=True)


# --------------------------------------------------------------------------------------
# weights helpers

def dense_to_groups(ob, W, names):
    clothing.write_groups(ob, rig.limit_normalize(W, 4), names)


def rigid(ob, bone, names):
    W = np.zeros((len(ob.data.vertices), len(names)), dtype=np.float32)
    W[:, names.index(bone)] = 1
    dense_to_groups(ob, W, names)


def from_source(ob, src, names):
    dense_to_groups(ob, clothing.weights_from(src, get_verts(ob), names), names)


def renormalize(ob, names):
    dense_to_groups(ob, rig.read_weights(ob, names), names)


def join(objs, name):
    objs = [o for o in objs if o is not None]
    set_active(objs[0])
    for o in objs[1:]:
        o.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    if len(objs) > 1:
        bpy.ops.object.join()
    ob = bpy.context.view_layer.objects.active
    ob.name = name
    ob.data.name = name
    return ob


def merge_close(ob, dist=1e-5):
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=dist)
    bm.to_mesh(ob.data)
    bm.free()


# --------------------------------------------------------------------------------------
# body culling, subdivision, blendshapes

def cull_hidden(B, garments, erode=3, reach=0.16):
    """Delete skin faces that are fully hidden under clothing (no poke-through, fewer tris)."""
    from mathutils import Vector
    co = get_verts(B)
    faces = [tuple(p.vertices) for p in B.data.polygons]
    nrm = geo.vertex_normals(co, faces)
    bvhs = [geo.bvh_from_object(g) for g in garments]
    covered = np.zeros(len(co), dtype=bool)
    for i in range(len(co)):
        o = Vector(co[i] + nrm[i] * 0.0005)
        d = Vector(nrm[i])
        for b in bvhs:
            hit = b.ray_cast(o, d, reach)
            if hit[0] is not None:
                covered[i] = True
                break
    A = geo.adjacency(faces, len(co))
    for _ in range(erode):
        nb_uncovered = (A @ (~covered).astype(float)) > 0
        covered &= ~nb_uncovered
    bm = bmesh.new()
    bm.from_mesh(B.data)
    dead = [f for f in bm.faces if all(covered[v.index] for v in f.verts)]
    bmesh.ops.delete(bm, geom=dead, context="FACES")
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context="VERTS")
    bm.to_mesh(B.data)
    bm.free()
    B.data.update()
    return covered


def subdivided_positions(src, co):
    """Positions of `src` topology with vertex positions `co`, after one Catmull-Clark level."""
    me = src.data.copy()
    tmp = bpy.data.objects.new("_tmp_sub", me)
    bpy.context.scene.collection.objects.link(tmp)
    tmp.modifiers.clear()
    tmp.vertex_groups.clear()
    set_verts(tmp, co)
    clothing.subdivide(tmp, 1)
    out = get_verts(tmp)
    bpy.data.objects.remove(tmp)
    bpy.data.meshes.remove(me)
    return out


UNIT_RACE = (("african", body.MACRO["african"]), ("asian", body.MACRO["asian"]), ("caucasian", body.MACRO["caucasian"]))


def expression_units():
    d = os.path.join(mh.MH_DIR, "targets", "expression", "units", "caucasian")
    return sorted(f[:-7] for f in os.listdir(d) if f.endswith(".target"))


def unit_delta(nverts, unit):
    tot = sum(w for _, w in UNIT_RACE)
    dm = np.zeros((nverts, 3))
    for race, w in UNIT_RACE:
        dm += mh.target_delta(nverts, f"expression/units/{race}/{unit}.target", w / tot)
    return mh.to_blender(dm)


def shape_name(unit):
    return "".join(p.capitalize() for p in unit.replace("_", "-").split("-"))


# --------------------------------------------------------------------------------------

def build(subdivide_body=True, with_shapes=True):
    butil.reset_scene()
    log("loading MakeHuman base + targets")
    base = mh.load_base_mesh()
    v = body.morph(base)
    objs, ids = body.build(base, v)
    B = objs["Body"]
    skel = mh.load_skeleton()
    arm = rig.build_armature(v, skel)
    names = [b.name for b in arm.data.bones]
    Wfull = rig.weight_matrix(rig.game_weights(skel, mh.load_weights()), names, len(base.verts))
    attr = B.data.attributes.new("mh_id", "INT", "POINT")
    attr.data.foreach_set("value", ids["Body"].astype(np.int32))
    B.data.update()

    log("demon arm sculpt")
    demon.sculpt(B, arm)
    full = B.copy()
    full.data = B.data.copy()
    full.name = "Body_Full"
    bpy.context.scene.collection.objects.link(full)

    log("garments")
    ctx = clothing.Ctx(base, v, B, arm, names, Wfull)
    tunic = clothing.tunic_upper(ctx)
    band = clothing.collar_band(ctx, tunic)
    cuffs = clothing.sleeve_cuffs(ctx, tunic)
    clothing.attach(ctx, [band] + cuffs, tunic)
    skirt = clothing.tunic_lower(ctx)
    pants = clothing.pants(ctx)
    belt, knot, tails = clothing.sash(ctx, [tunic, skirt], [skirt, pants])
    wraps = clothing.leg_wraps(ctx)
    straps = clothing.wrap_straps(ctx, wraps)
    shoe_parts = clothing.shoes(ctx)

    log("claws / spikes")
    claws = demon.claws(arm)
    spikes = demon.spikes(arm, B)

    log("hair")
    cap = hair.scalp_cap(B)
    cards, info, strands = hair.build_cards(B, [B, tunic])
    fly = hair.flyaways(B)
    brows = hair.brows(B)
    lashes, lash_ids = hair.lashes(base, v)
    chains = hair.add_hair_bones(arm, strands)
    old_names = names
    names = [b.name for b in arm.data.bones]          # bone order changes when bones are added
    W2 = np.zeros((len(Wfull), len(names)), np.float32)
    for j, n in enumerate(old_names):
        W2[:, names.index(n)] = Wfull[:, j]
    Wfull = W2
    ctx.bone_names = names

    log("skin weights")
    for n in ("Body", "Teeth", "Tongue"):
        dense_to_groups(objs[n], Wfull[ids[n]], names)
    dense_to_groups(full, Wfull[ids["Body"]], names)
    eo = objs["Eyes"]
    We = np.zeros((len(eo.data.vertices), len(names)), np.float32)
    ec = get_verts(eo)
    We[ec[:, 0] > 0, names.index("eye_l")] = 1
    We[ec[:, 0] <= 0, names.index("eye_r")] = 1
    dense_to_groups(eo, We, names)
    dense_to_groups(lashes, Wfull[lash_ids], names)
    for o in (brows, cap, fly):
        rigid(o, "head", names)
    hc = get_verts(cards)
    blend = smoothstep(1.83, 1.75, hc[:, 2]) * (hc[:, 1] > hair.HC[1] + 0.03)
    dense_to_groups(cards, hair.hair_weights(hc, chains, names, blend), names)
    for o in (tunic, skirt, pants, wraps):
        renormalize(o, names)
    for o in [belt] + knot:
        from_source(o, skirt, names)
    for o in tails:
        from_source(o, skirt, names)
    for o in straps:
        from_source(o, wraps, names)
    for o in shoe_parts:
        renormalize(o, names)
    for c in claws:
        f = c.name.split("_")[1]
        rigid(c, f"{f}_03_l", names)
    for s in spikes:
        from_source(s, B, names)

    log("culling hidden skin")
    garments = [tunic, skirt, pants, wraps, belt] + [o for o in shoe_parts if o.name.startswith(("Shoe_", "Sole_"))]
    cull_hidden(B, garments)
    pre_ids = np.zeros(len(B.data.vertices), dtype=np.int32)
    B.data.attributes["mh_id"].data.foreach_get("value", pre_ids)
    pre_co = get_verts(B)
    pre = B.copy()
    pre.data = B.data.copy()
    bpy.context.scene.collection.objects.link(pre)

    if subdivide_body:
        log("subdividing visible skin")
        clothing.subdivide(B, 1)
        renormalize(B, names)

    if with_shapes:
        log("facial blendshapes (MakeHuman expression units)")
        B.shape_key_add(name="Basis")
        lashes.shape_key_add(name="Basis")
        brows.shape_key_add(name="Basis")
        bco = get_verts(B)
        lco = get_verts(lashes)
        brco = get_verts(brows)
        tree = cKDTree(bco)
        _, br_nn = tree.query(brco)
        for unit in expression_units():
            d = unit_delta(len(base.verts), unit)
            if np.abs(d[pre_ids]).max() < 1e-6:
                continue
            co_k = pre_co + d[pre_ids]
            sub = subdivided_positions(pre, co_k) if subdivide_body else co_k
            nm = shape_name(unit)
            k = B.shape_key_add(name=nm, from_mix=False)
            k.data.foreach_set("co", sub.ravel())
            k2 = lashes.shape_key_add(name=nm, from_mix=False)
            k2.data.foreach_set("co", (lco + d[lash_ids]).ravel())
            k3 = brows.shape_key_add(name=nm, from_mix=False)
            k3.data.foreach_set("co", (brco + (sub - bco)[br_nn]).ravel())
            for kk in (k, k2, k3):
                kk.value = 0.0
    bpy.data.objects.remove(pre)
    B.data.attributes.remove(B.data.attributes["mh_id"])
    full.data.attributes.remove(full.data.attributes["mh_id"])

    for o in claws + spikes:
        merge_close(o, 1e-6)

    parts = dict(
        body=B, body_full=full, eyes=objs["Eyes"], teeth=objs["Teeth"], tongue=objs["Tongue"],
        lashes=lashes, brows=brows, hair_cap=cap, hair_cards=cards, hair_fly=fly,
        tunic=tunic, skirt=skirt, pants=pants, wraps=wraps, belt=belt, knot=knot, tails=tails,
        straps=straps, shoes=shoe_parts, claws=claws, spikes=spikes, arm=arm, names=names,
    )
    log("geometry + rig done")
    return parts
