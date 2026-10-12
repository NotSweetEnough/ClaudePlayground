"""MakeHuman (hm08) data access: base mesh, morph targets, skeleton joints, proxies.

All MakeHuman data used here is CC0 (see third_party/makehuman/LICENSE.md).
Coordinates are kept in MakeHuman space (Y-up, decimetres) until `to_blender`.
"""
import json
import os
from dataclasses import dataclass, field

import numpy as np

from .paths import MH_DIR


@dataclass
class BaseMesh:
    verts: np.ndarray            # (N, 3) MakeHuman space
    uvs: np.ndarray              # (M, 2)
    faces: np.ndarray            # (F, 4) vertex indices (hm08 is all quads)
    face_uvs: np.ndarray         # (F, 4) uv indices
    face_group: np.ndarray       # (F,) group index
    groups: list = field(default_factory=list)

    def group_faces(self, name):
        gi = self.groups.index(name)
        return np.nonzero(self.face_group == gi)[0]

    def group_verts(self, name):
        return np.unique(self.faces[self.group_faces(name)].ravel())

    def groups_matching(self, prefix):
        return [g for g in self.groups if g.startswith(prefix)]


def load_base_mesh(path=None):
    path = path or os.path.join(MH_DIR, "3dobjs", "base.obj")
    verts, uvs, faces, fuvs, fgroup, groups = [], [], [], [], [], []
    gi = -1
    with open(path) as fh:
        for line in fh:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("vt "):
                uvs.append([float(x) for x in line.split()[1:3]])
            elif line.startswith("g "):
                name = line.split(None, 1)[1].strip()
                if name in groups:
                    gi = groups.index(name)
                else:
                    groups.append(name)
                    gi = len(groups) - 1
            elif line.startswith("f "):
                vi, ti = [], []
                for tok in line.split()[1:]:
                    p = tok.split("/")
                    vi.append(int(p[0]) - 1)
                    ti.append(int(p[1]) - 1 if len(p) > 1 and p[1] else -1)
                faces.append(vi)
                fuvs.append(ti)
                fgroup.append(gi)
    return BaseMesh(np.array(verts, dtype=np.float64), np.array(uvs, dtype=np.float64),
                    np.array(faces, dtype=np.int64), np.array(fuvs, dtype=np.int64),
                    np.array(fgroup, dtype=np.int64), groups)


def load_target(rel):
    """Return (indices, deltas) of a .target file (path relative to data/targets)."""
    path = os.path.join(MH_DIR, "targets", rel)
    idx, d = [], []
    with open(path) as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            p = line.split()
            idx.append(int(p[0]))
            d.append([float(p[1]), float(p[2]), float(p[3])])
    if not idx:
        return np.zeros(0, dtype=np.int64), np.zeros((0, 3))
    return np.array(idx, dtype=np.int64), np.array(d)


def apply_targets(verts, targets):
    out = verts.copy()
    for rel, w in targets:
        if abs(w) < 1e-6:
            continue
        idx, d = load_target(rel)
        if len(idx):
            np.add.at(out, idx, d * w)
    return out


def target_delta(nverts, rel, w=1.0):
    out = np.zeros((nverts, 3))
    idx, d = load_target(rel)
    if len(idx):
        out[idx] = d * w
    return out


# --- MakeHuman macro modifier weighting (mirrors apps/human.py) -----------------

def _age_vals(age):
    if age < 0.5:
        young = max(0.0, (age - 0.1875) * 3.2)
        baby = max(0.0, 1 - age * 5.333)
        child = max(0.0, min(1.0, 5.333 * age) - young)
        return {"baby": baby, "child": child, "young": young, "old": 0.0}
    old = max(0.0, age * 2 - 1)
    return {"baby": 0.0, "child": 0.0, "young": 1 - old, "old": old}


def _tri(v, lo, mid, hi):
    if v < 0.5:
        a = max(0.0, 1 - v * 2)
        return {lo: a, mid: 1 - a, hi: 0.0}
    b = max(0.0, v * 2 - 1)
    return {lo: 0.0, mid: 1 - b, hi: b}


def macro_targets(gender=0.5, age=0.5, muscle=0.5, weight=0.5, height=0.5,
                  proportions=0.5, african=1 / 3, asian=1 / 3, caucasian=1 / 3):
    g = {"female": 1 - gender, "male": gender}
    a = _age_vals(age)
    m = _tri(muscle, "minmuscle", "averagemuscle", "maxmuscle")
    w = _tri(weight, "minweight", "averageweight", "maxweight")
    h = _tri(height, "minheight", "averageheight", "maxheight")
    p = _tri(proportions, "uncommonproportions", "regularproportions", "idealproportions")
    tot = african + asian + caucasian
    race = {"african": african / tot, "asian": asian / tot, "caucasian": caucasian / tot}
    out = []
    for gk, gv in g.items():
        for ak, av in a.items():
            if gv * av == 0:
                continue
            for rk, rv in race.items():
                out.append((f"macrodetails/{rk}-{gk}-{ak}.target", gv * av * rv))
            for mk, mv in m.items():
                for wk, wv in w.items():
                    base = gv * av * mv * wv
                    if base == 0:
                        continue
                    out.append((f"macrodetails/universal-{gk}-{ak}-{mk}-{wk}.target", base))
                    for hk, hv in h.items():
                        if hk != "averageheight" and hv:
                            out.append((f"macrodetails/height/{gk}-{ak}-{mk}-{wk}-{hk}.target", base * hv))
                    for pk, pv in p.items():
                        if pk != "regularproportions" and pv:
                            out.append((f"macrodetails/proportions/{gk}-{ak}-{mk}-{wk}-{pk}.target", base * pv))
    return out


def modifier_targets(mods):
    """mods: {"armslegs/l-lowerarm-muscle-decr|incr": 0.6, "torso/torso-vshape-decr|incr": .5, "head/head-square": .3}"""
    out = []
    for key, v in mods.items():
        if "|" in key:
            folder, rest = key.split("/", 1)
            stem, pair = rest.rsplit("-", 1)
            lo, hi = pair.split("|")
            name = hi if v >= 0 else lo
            out.append((f"{folder}/{stem}-{name}.target", abs(v)))
        else:
            out.append((f"{key}.target", v))
    return out


# --- skeleton -----------------------------------------------------------------

def load_skeleton():
    with open(os.path.join(MH_DIR, "rigs", "default.mhskel")) as fh:
        return json.load(fh)


def load_weights():
    with open(os.path.join(MH_DIR, "rigs", "default_weights.mhw")) as fh:
        return json.load(fh)["weights"]


def joint_positions(verts, skel):
    return {name: verts[idx].mean(axis=0) for name, idx in skel["joints"].items()}


# --- proxies (mhclo) ------------------------------------------------------------

def fit_proxy(mhclo_path, verts):
    """Fit a MakeHuman proxy (.mhclo + .obj) to `verts`. Returns (proxy_verts, obj_path)."""
    scales = {}
    refs = []
    obj_file = None
    in_verts = False
    with open(mhclo_path) as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            p = s.split()
            if in_verts:
                if p[0] in ("delete_verts", "material") or not p[0].lstrip("-").isdigit():
                    in_verts = False
                    continue
                refs.append(p)
                continue
            if p[0] in ("x_scale", "y_scale", "z_scale"):
                scales[p[0][0]] = (int(p[1]), int(p[2]), float(p[3]))
            elif p[0] == "obj_file":
                obj_file = os.path.join(os.path.dirname(mhclo_path), p[1])
            elif p[0] == "verts":
                in_verts = True
    sc = np.ones(3)
    for i, ax in enumerate("xyz"):
        if ax in scales:
            a, b, ref = scales[ax]
            sc[i] = abs(verts[a][i] - verts[b][i]) / ref
    out = []
    for p in refs:
        if len(p) == 1:
            out.append(verts[int(p[0])])
        else:
            v = [int(x) for x in p[:3]]
            w = np.array([float(x) for x in p[3:6]])
            off = np.array([float(x) for x in p[6:9]])
            out.append((verts[v] * w[:, None]).sum(0) + off * sc)
    return np.array(out), obj_file


def load_simple_obj(path):
    verts, uvs, faces, fuvs = [], [], [], []
    with open(path) as fh:
        for line in fh:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("vt "):
                uvs.append([float(x) for x in line.split()[1:3]])
            elif line.startswith("f "):
                vi, ti = [], []
                for tok in line.split()[1:]:
                    p = tok.split("/")
                    vi.append(int(p[0]) - 1)
                    ti.append(int(p[1]) - 1 if len(p) > 1 and p[1] else -1)
                faces.append(vi)
                fuvs.append(ti)
    return np.array(verts), np.array(uvs), faces, fuvs


def to_blender(v):
    """MakeHuman (x, y-up, z-forward; dm) -> Blender (x, -y forward, z-up; m)."""
    v = np.asarray(v, dtype=np.float64)
    out = np.empty_like(v)
    out[..., 0] = v[..., 0] * 0.1
    out[..., 1] = -v[..., 2] * 0.1
    out[..., 2] = v[..., 1] * 0.1
    return out
