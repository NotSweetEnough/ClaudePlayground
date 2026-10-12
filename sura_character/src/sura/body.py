"""Body: MakeHuman hm08 base mesh morphed into the Sura's proportions."""
import os

import numpy as np

from . import mh
from .butil import mesh_object

# Lean, tall, battle-hardened male; mixed East-Asian / European features.
MACRO = dict(gender=1.0, age=0.56, muscle=0.80, weight=0.40, height=0.66,
             proportions=0.9, african=0.05, asian=0.45, caucasian=0.50)

MODIFIERS = {
    # skull / face: gaunt, angular, brooding
    "head/head-square": 0.20,
    "head/head-fat-decr|incr": -0.45,
    "head/head-age-decr|incr": 0.15,
    "cheek/l-cheek-bones-decr|incr": 0.55,
    "cheek/r-cheek-bones-decr|incr": 0.55,
    "cheek/l-cheek-volume-decr|incr": -0.55,
    "cheek/r-cheek-volume-decr|incr": -0.55,
    "cheek/l-cheek-inner-decr|incr": -0.3,
    "cheek/r-cheek-inner-decr|incr": -0.3,
    "chin/chin-prominent-decr|incr": 0.35,
    "chin/chin-width-decr|incr": 0.15,
    "chin/chin-bones-decr|incr": 0.3,
    "eyebrows/eyebrows-trans-down|up": -0.35,
    "eyebrows/eyebrows-angle-down|up": 0.45,
    "forehead/forehead-temple-decr|incr": -0.3,
    "nose/nose-hump-decr|incr": 0.35,
    "nose/nose-point-width-decr|incr": -0.35,
    "nose/nose-scale-vert-decr|incr": 0.15,
    "nose/nose-nostrils-width-decr|incr": -0.2,
    "mouth/mouth-laugh-lines-in|out": 0.6,
    "mouth/mouth-upperlip-volume-decr|incr": -0.35,
    "mouth/mouth-lowerlip-volume-decr|incr": -0.15,
    "mouth/mouth-angles-down|up": -0.25,
    "eyes/l-eye-height2-decr|incr": -0.25,
    "eyes/r-eye-height2-decr|incr": -0.25,
    "eyes/l-eye-corner2-down|up": 0.3,
    "eyes/r-eye-corner2-down|up": 0.3,
    "eyes/l-eye-bag-decr|incr": 0.25,
    "eyes/r-eye-bag-decr|incr": 0.25,
    "ears/l-ear-shape-pointed": 0.3,
    "ears/r-ear-shape-pointed": 0.3,
    "neck/neck-scale-horiz-decr|incr": 0.25,
    "neck/neck-scale-depth-decr|incr": 0.15,
    # torso: V-shape warrior build
    "torso/torso-vshape-decr|incr": 0.55,
    "torso/torso-muscle-dorsi-decr|incr": 0.45,
    "torso/torso-muscle-pectoral-decr|incr": 0.35,
    "armslegs/r-upperarm-muscle-decr|incr": 0.35,
    "armslegs/r-upperarm-shoulder-muscle-decr|incr": 0.35,
    "armslegs/l-upperarm-shoulder-muscle-decr|incr": 0.45,
    "armslegs/r-lowerarm-muscle-decr|incr": 0.3,
    # demon left arm: swollen forearm, large long-fingered hand
    "armslegs/l-upperarm-muscle-decr|incr": 0.75,
    "armslegs/l-lowerarm-muscle-decr|incr": 1.0,
    "armslegs/l-lowerarm-scale-horiz-decr|incr": 0.45,
    "armslegs/l-lowerarm-scale-depth-decr|incr": 0.45,
    "armslegs/l-hand-scale-decr|incr": 0.55,
    "armslegs/l-hand-fingers-length-decr|incr": 0.9,
    "armslegs/l-hand-fingers-diameter-decr|incr": 0.35,
    "armslegs/l-upperleg-muscle-decr|incr": 0.3,
    "armslegs/r-upperleg-muscle-decr|incr": 0.3,
    "armslegs/l-lowerleg-muscle-decr|incr": 0.35,
    "armslegs/r-lowerleg-muscle-decr|incr": 0.35,
}

SOLE = 0.024   # shoe sole thickness (m): the body is lifted so the shoes stand on z=0

# Pieces of the base mesh we keep as real geometry.
KEEP = {
    "Body": ["body"],
    "Teeth": ["helper-upper-teeth", "helper-lower-teeth"],
    "Tongue": ["helper-tongue"],
}


def all_targets():
    return mh.macro_targets(**MACRO) + mh.modifier_targets(MODIFIERS)


def morph(base):
    """Morphed hm08 vertices (MakeHuman space), translated so the soles rest on y=0."""
    v = mh.apply_targets(base.verts, all_targets())
    v[:, 1] -= v[base.group_verts("body"), 1].min() - SOLE * 10.0
    return v


def submesh(base, verts_mh, groups):
    """Extract faces of `groups` -> (blender verts, faces, per-face uvs, original vertex ids)."""
    fidx = np.concatenate([base.group_faces(g) for g in groups])
    faces = base.faces[fidx]
    used = np.unique(faces.ravel())
    remap = -np.ones(len(base.verts), dtype=np.int64)
    remap[used] = np.arange(len(used))
    uvs = [[tuple(base.uvs[t]) for t in base.face_uvs[f]] for f in fidx]
    return mh.to_blender(verts_mh[used]), remap[faces], uvs, used


def build_eyes(verts_mh, collection=None):
    pv, obj_path = mh.fit_proxy(os.path.join(mh.MH_DIR, "eyes", "high-poly", "high-poly.mhclo"), verts_mh)
    _, puv, pf, pfuv = mh.load_simple_obj(obj_path)
    uvs = [[tuple(puv[t]) for t in fu] for fu in pfuv]
    return mesh_object("Eyes", mh.to_blender(pv), pf, uvs, collection)


def build(base, verts_mh, collection=None):
    objs, ids = {}, {}
    for name, groups in KEEP.items():
        v, f, uv, used = submesh(base, verts_mh, groups)
        objs[name] = mesh_object(name, v, f, uv, collection)
        ids[name] = used
    objs["Eyes"] = build_eyes(verts_mh, collection)
    return objs, ids
