"""Game skeleton (Unreal-mannequin naming) derived from the MakeHuman default rig.

The 163-bone MakeHuman skeleton (with its CC0 skin weights) is collapsed into a
~80 bone game skeleton: body + fingers + twist + jaw/eyes. Weights of merged
MakeHuman bones are summed, then limited to 4 influences per vertex.
"""
import numpy as np
import bpy
from mathutils import Vector

from . import mh
from .butil import set_active

# name, parent, head, tail, roll plane (MakeHuman bone), MakeHuman source bones
# head/tail: MakeHuman joint names, or tuples handled in _joint().
_SIDE = [
    ("clavicle_{s}", "spine_05", "clavicle.{S}____head", "shoulder01.{S}____tail", "clavicle.{S}", ["clavicle.{S}", "shoulder01.{S}"]),
    ("upperarm_{s}", "clavicle_{s}", "upperarm01.{S}____head", "upperarm02.{S}____tail", "upperarm02.{S}", ["upperarm01.{S}", "upperarm02.{S}"]),
    ("lowerarm_{s}", "upperarm_{s}", "lowerarm01.{S}____head", "lowerarm02.{S}____tail", "lowerarm01.{S}", ["lowerarm01.{S}"]),
    ("lowerarm_twist_01_{s}", "lowerarm_{s}", "lowerarm02.{S}____head", "lowerarm02.{S}____tail", "lowerarm02.{S}", ["lowerarm02.{S}"]),
    ("hand_{s}", "lowerarm_{s}", "wrist.{S}____head", ("mid", "metacarpal2.{S}____head", "metacarpal2.{S}____tail"), "wrist.{S}", ["wrist.{S}"]),
    ("thumb_01_{s}", "hand_{s}", "finger1-1.{S}____head", "finger1-1.{S}____tail", "finger1-1.{S}", ["finger1-1.{S}"]),
    ("thumb_02_{s}", "thumb_01_{s}", "finger1-2.{S}____head", "finger1-2.{S}____tail", "finger1-2.{S}", ["finger1-2.{S}"]),
    ("thumb_03_{s}", "thumb_02_{s}", "finger1-3.{S}____head", "finger1-3.{S}____tail", "finger1-3.{S}", ["finger1-3.{S}"]),
]
for _i, _f in enumerate(["index", "middle", "ring", "pinky"]):
    _m, _n = _i + 1, _i + 2
    _SIDE.append((f"{_f}_metacarpal_{{s}}", "hand_{s}", f"metacarpal{_m}.{{S}}____head", f"metacarpal{_m}.{{S}}____tail",
                  f"metacarpal{_m}.{{S}}", [f"metacarpal{_m}.{{S}}"]))
    for _k in (1, 2, 3):
        _par = f"{_f}_metacarpal_{{s}}" if _k == 1 else f"{_f}_{_k - 1:02d}_{{s}}"
        _SIDE.append((f"{_f}_{_k:02d}_{{s}}", _par, f"finger{_n}-{_k}.{{S}}____head", f"finger{_n}-{_k}.{{S}}____tail",
                      f"finger{_n}-{_k}.{{S}}", [f"finger{_n}-{_k}.{{S}}"]))
_SIDE += [
    ("thigh_{s}", "pelvis", "upperleg01.{S}____head", "upperleg02.{S}____tail", "upperleg02.{S}", ["upperleg01.{S}", "upperleg02.{S}", "pelvis.{S}"]),
    ("calf_{s}", "thigh_{s}", "lowerleg01.{S}____head", "lowerleg02.{S}____tail", "lowerleg01.{S}", ["lowerleg01.{S}"]),
    ("calf_twist_01_{s}", "calf_{s}", "lowerleg02.{S}____head", "lowerleg02.{S}____tail", "lowerleg02.{S}", ["lowerleg02.{S}"]),
    ("foot_{s}", "calf_{s}", "foot.{S}____head", ("avg", "toe2-1.{S}____head", "toe3-1.{S}____head", "toe4-1.{S}____head"), "foot.{S}", ["foot.{S}"]),
    ("ball_{s}", "foot_{s}", ("avg", "toe2-1.{S}____head", "toe3-1.{S}____head", "toe4-1.{S}____head"),
     ("avg", "toe2-3.{S}____tail", "toe3-3.{S}____tail", "toe4-3.{S}____tail"), "foot.{S}",
     ["toe1-1.{S}", "toe1-2.{S}"] + [f"toe{t}-{k}.{{S}}" for t in (2, 3, 4, 5) for k in (1, 2, 3)]),
]

_JAW_TREE = ["jaw", "special04", "oris02", "oris01", "oris06.L", "oris06.R", "oris07.L", "oris07.R",
             "tongue00", "tongue01", "tongue02", "tongue03", "tongue04", "tongue05.L", "tongue05.R",
             "tongue06.L", "tongue06.R", "tongue07.L", "tongue07.R"]

_CENTER = [
    ("root", None, ("abs", (0, 0, 0)), ("abs", (0, 0, 0.25)), None, []),
    ("pelvis", "root", ("pelvis_head",), "spine05____head", "spine05", ["root"]),
    ("spine_01", "pelvis", "spine05____head", "spine05____tail", "spine05", ["spine05"]),
    ("spine_02", "spine_01", "spine04____head", "spine04____tail", "spine04", ["spine04"]),
    ("spine_03", "spine_02", "spine03____head", "spine03____tail", "spine03", ["spine03"]),
    ("spine_04", "spine_03", "spine02____head", "spine02____tail", "spine02", ["spine02", "breast.L", "breast.R"]),
    ("spine_05", "spine_04", "spine01____head", "spine01____tail", "spine01", ["spine01"]),
    ("neck_01", "spine_05", "neck01____head", "neck01____tail", "neck01", ["neck01"]),
    ("neck_02", "neck_01", "neck02____head", "neck03____tail", "neck02", ["neck02", "neck03"]),
    ("head", "neck_02", "head____head", "head____tail", "head", None),  # None = everything left over in the head
    ("jaw", "head", "jaw____head", "jaw____tail", "jaw", _JAW_TREE),
    ("eye_l", "head", "eye.L____head", "eye.L____tail", "eye.L", ["eye.L"]),
    ("eye_r", "head", "eye.R____head", "eye.R____tail", "eye.R", ["eye.R"]),
]


def bone_table():
    table = list(_CENTER)
    for s, S in (("l", "L"), ("r", "R")):
        for name, par, h, t, plane, src in _SIDE:
            f = lambda x: x.format(s=s, S=S) if isinstance(x, str) else tuple(f(y) for y in x) if isinstance(x, tuple) else x
            table.append((f(name), f(par), f(h), f(t), f(plane), [f(x) for x in src]))
    return table


def _joint(J, spec):
    if isinstance(spec, str):
        return J[spec]
    kind = spec[0]
    if kind == "abs":
        return np.array(spec[1], dtype=float)
    if kind in ("avg", "mid"):
        return np.mean([J[x] for x in spec[1:]], axis=0)
    if kind == "pelvis_head":
        hip = (J["upperleg01.L____head"] + J["upperleg01.R____head"]) / 2
        p = J["spine05____head"].copy()
        p[2] = hip[2] - 0.06
        return p
    raise ValueError(spec)


def mh_to_game_map(skel):
    """MakeHuman bone -> game bone."""
    table = bone_table()
    m = {}
    for name, _, _, _, _, src in table:
        for b in src or []:
            m[b] = name
    for b in skel["bones"]:
        m.setdefault(b, "head")  # remaining facial bones collapse into the head
    return m


def joints_blender(verts_mh, skel):
    J = mh.joint_positions(verts_mh, skel)
    return {k: mh.to_blender(v) for k, v in J.items()}


def build_armature(verts_mh, skel, name="Sura_Rig", collection=None):
    J = joints_blender(verts_mh, skel)
    table = bone_table()
    arm = bpy.data.armatures.new(name)
    arm.display_type = "OCTAHEDRAL"
    ob = bpy.data.objects.new(name, arm)
    (collection or bpy.context.scene.collection).objects.link(ob)
    ob.show_in_front = True
    set_active(ob)
    bpy.ops.object.mode_set(mode="EDIT")
    eb = arm.edit_bones
    for bname, par, h, t, plane, _ in table:
        b = eb.new(bname)
        b.head = Vector(_joint(J, h))
        b.tail = Vector(_joint(J, t))
        if (b.tail - b.head).length < 1e-3:
            b.tail = b.head + Vector((0, 0, 0.02))
        b.use_deform = True
    for bname, par, h, t, plane, _ in table:
        if par:
            eb[bname].parent = eb[par]
    # rolls: local X = normal of the MakeHuman rotation plane (bend axis)
    planes = skel["planes"]
    for bname, par, h, t, plane, _ in table:
        b = eb[bname]
        y = (b.tail - b.head).normalized()
        if plane is None:
            x = Vector((1, 0, 0))
        else:
            p = [Vector(J[j]) for j in planes[plane + "____plane"]]
            x = (p[1] - p[0]).cross(p[2] - p[1])
            if x.length < 1e-9:
                x = Vector((1, 0, 0))
        x = (x - x.dot(y) * y)
        if x.length < 1e-6:
            x = Vector((1, 0, 0)) - Vector((1, 0, 0)).dot(y) * y
        x.normalize()
        b.align_roll(x.cross(y))
    # spine/neck/head/root rolls: keep X lateral (+X)
    for bname in ["root", "pelvis", "spine_01", "spine_02", "spine_03", "spine_04", "spine_05",
                  "neck_01", "neck_02", "head", "jaw", "eye_l", "eye_r"]:
        b = eb[bname]
        y = (b.tail - b.head).normalized()
        x = Vector((1, 0, 0))
        x = (x - x.dot(y) * y).normalized()
        b.align_roll(x.cross(y))
    bpy.ops.object.mode_set(mode="OBJECT")
    return ob


def game_weights(skel, mh_weights):
    """Return {game_bone: (vertex_ids, weights)} over MakeHuman vertex ids (summed)."""
    m = mh_to_game_map(skel)
    acc = {}
    for b, lst in mh_weights.items():
        g = m.get(b, "head")
        d = acc.setdefault(g, {})
        for vid, w in lst:
            d[vid] = d.get(vid, 0.0) + w
    return acc


def weight_matrix(acc, bone_names, nverts):
    W = np.zeros((nverts, len(bone_names)), dtype=np.float32)
    idx = {b: i for i, b in enumerate(bone_names)}
    for b, d in acc.items():
        if b not in idx:
            continue
        ids = np.fromiter(d.keys(), dtype=np.int64)
        ws = np.fromiter(d.values(), dtype=np.float32)
        W[ids, idx[b]] += ws
    return W


def limit_normalize(W, k=4):
    W = W.copy()
    if W.shape[1] > k:
        part = np.argpartition(-W, k, axis=1)[:, k:]
        np.put_along_axis(W, part, 0.0, axis=1)
    W[W < 0.01] = 0.0
    s = W.sum(1, keepdims=True)
    s[s == 0] = 1
    return W / s


def assign_weights(ob, W, bone_names, arm_ob):
    """Create vertex groups from dense weight matrix (rows match ob vertices)."""
    ob.vertex_groups.clear()
    for j, b in enumerate(bone_names):
        col = W[:, j]
        nz = np.nonzero(col > 0)[0]
        if not len(nz):
            continue
        vg = ob.vertex_groups.new(name=b)
        for w in np.unique(np.round(col[nz], 4)):
            sel = nz[np.round(col[nz], 4) == w]
            vg.add(sel.tolist(), float(w), "REPLACE")
    mod = ob.modifiers.new("Armature", "ARMATURE")
    mod.object = arm_ob
    ob.parent = arm_ob


def read_weights(ob, bone_names):
    """Dense (nverts, nbones) matrix from an object's vertex groups."""
    n = len(ob.data.vertices)
    W = np.zeros((n, len(bone_names)), dtype=np.float32)
    gi = {vg.index: bone_names.index(vg.name) for vg in ob.vertex_groups if vg.name in bone_names}
    for v in ob.data.vertices:
        for g in v.groups:
            if g.group in gi:
                W[v.index, gi[g.group]] = g.weight
    return W
