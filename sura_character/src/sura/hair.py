"""Long, slicked-back grey hair: scalp cap + layered alpha cards + hair bone chains.

Strands are traced across the scalp toward an attractor behind the head (slick
back), then fall under gravity down the back with collision against body/tunic.
"""
import math

import bpy
import numpy as np
from mathutils import Vector

from . import geo
from .butil import get_verts, mesh_object, set_active, smoothstep

HC = np.array([0.0, -0.055, 1.86])          # head centre (rest pose)
ATTR = np.array([0.0, 0.40, 1.55])          # slick-back attractor behind / below the head
N_COLS = 6                                   # atlas columns: 4 hair card variants, lashes/brows, opaque scalp
N_HAIR_COLS = 4
LASH_COL = 4
CAP_COL = 5


def scalp_mask(co):
    rel = co - HC
    rxy = np.maximum(np.hypot(rel[:, 0], rel[:, 1]), 1e-6)
    f = -rel[:, 1] / rxy
    zline = np.interp(f, [-1.0, -0.35, -0.05, 0.25, 0.6, 1.0], [1.762, 1.792, 1.845, 1.858, 1.884, 1.902])
    zline -= 0.006 * np.exp(-(co[:, 0] / 0.018) ** 2) * (f > 0.8)          # slight widow's peak
    m = smoothstep(zline - 0.003, zline + 0.006, co[:, 2])
    ear = (np.abs(co[:, 0]) > 0.072) & (co[:, 2] < 1.878) & (co[:, 1] > -0.085) & (co[:, 1] < 0.015)
    return m * (co[:, 2] > 1.74) * (np.abs(co[:, 0]) < 0.12) * ~ear


# --------------------------------------------------------------------------------------

def scalp_cap(body, offset=0.0015):
    import bmesh
    co = get_verts(body)
    m = scalp_mask(co) > 0.3
    bm = bmesh.new()
    bm.from_mesh(body.data)
    keep = [f for f in bm.faces if all(m[v.index] for v in f.verts)]
    drop = [f for f in bm.faces if f not in set(keep)]
    bmesh.ops.delete(bm, geom=drop, context="FACES")
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context="VERTS")
    me = bpy.data.meshes.new("HairCap")
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new("HairCap", me)
    bpy.context.scene.collection.objects.link(ob)
    ob.vertex_groups.clear()
    for l in list(me.uv_layers):
        me.uv_layers.remove(l)
    c = get_verts(ob)
    nrm = geo.vertex_normals(c, [tuple(p.vertices) for p in me.polygons])
    from .butil import set_verts
    set_verts(ob, c + nrm * offset)
    # UV: strands run front -> back over the skull
    c = get_verts(ob)
    rel = c - HC
    phi = np.arctan2(rel[:, 2], -rel[:, 1])                  # 0 at the forehead, pi at the nape side
    uvl = me.uv_layers.new(name="UVMap")
    col = CAP_COL
    for poly in me.polygons:
        for li in poly.loop_indices:
            vi = me.loops[li].vertex_index
            u = (col + 0.5 + 0.45 * np.clip(rel[vi, 0] / 0.1, -1, 1)) / N_COLS
            v = 1.0 - np.clip(phi[vi] / 2.6, 0, 1) * 0.55
            uvl.data[li].uv = (u, v)
    me.shade_smooth()
    return ob


# --------------------------------------------------------------------------------------

def _sample_roots(body, n, min_d, rng, region=None, hairline=False):
    co = get_verts(body)
    me = body.data
    m = scalp_mask(co)
    tris = []
    for p in me.polygons:
        vs = list(p.vertices)
        mv = m[vs]
        if (max(mv) > 0.45 and min(mv) < 0.85) if hairline else min(mv) > 0.35:
            for k in range(1, len(vs) - 1):
                tris.append((vs[0], vs[k], vs[k + 1]))
    tris = np.array(tris)
    a, b, c = co[tris[:, 0]], co[tris[:, 1]], co[tris[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    pick = rng.choice(len(tris), size=n * 8, p=area / area.sum())
    r1, r2 = rng.rand(len(pick)), rng.rand(len(pick))
    s = np.sqrt(r1)
    pts = a[pick] * (1 - s)[:, None] + b[pick] * (s * (1 - r2))[:, None] + c[pick] * (s * r2)[:, None]
    if region is not None:
        pts = pts[region(pts)]
    out = []
    for p in pts:
        if all(np.linalg.norm(p - q) > min_d for q in out):
            out.append(p)
            if len(out) >= n:
                break
    return np.array(out)


def trace(root, h, z_end, head_bvh, coll_bvhs, rng, step=0.008, max_len=None):
    """Returns (points, normals, phase) with phase 0 on the scalp and 1 hanging."""
    pts, nrms, phase = [], [], []
    loc, n, _, _ = head_bvh.find_nearest(Vector(root))
    p = np.array(loc) + np.array(n) * h
    d_prev = None
    hanging = False
    jitter = rng.normal(0, 0.15, 3)
    sway, phase0 = rng.uniform(0.05, 0.16), rng.uniform(0, 6.28)
    for it in range(400):
        if max_len is not None and it * step > max_len:
            break
        if not hanging:
            loc, n, _, _ = head_bvh.find_nearest(Vector(p))
            n = np.array(n)
            # in front of the crown aim back-and-up over the skull, behind it aim back-and-down
            front = 1.0 - smoothstep(HC[1] - 0.02, HC[1] + 0.05, p[1])
            target = ATTR + np.array([0.0, 0.0, 0.5 * front])
            d = target + jitter * [0.15, 0.05, 0.05] - p
            d -= (d @ n) * n
            if front > 0.5 and d[2] < 0:
                d[2] = -d[2]
            d /= np.linalg.norm(d)
            if d_prev is not None:
                d = d * 0.45 + d_prev * 0.55
                d /= np.linalg.norm(d)
            q = p + d * step
            loc2, n2, _, _ = head_bvh.find_nearest(Vector(q))
            n2 = np.array(n2)
            q = np.array(loc2) + n2 * h
            pts.append(p), nrms.append(n), phase.append(0.0)
            if n2[2] < -0.15 and q[1] > HC[1] + 0.075:
                hanging = True
                d_prev = (q - p) / max(np.linalg.norm(q - p), 1e-6)
            else:
                d_prev = (q - p) / max(np.linalg.norm(q - p), 1e-6)
            p = q
        else:
            fall = HC[2] - p[2]
            back = 0.55 * math.exp(-max(fall, 0) / 0.08) + 0.12
            grav = np.array([-p[0] * (1.2 if fall < 0.15 else 0.12) + sway * math.sin(fall * 9 + phase0), back, -1.0])
            grav /= np.linalg.norm(grav)
            d = d_prev * 0.55 + grav * 0.45
            d /= np.linalg.norm(d)
            q = p + d * step
            nb = None
            for bvh in coll_bvhs:
                loc, nn, _, dist = bvh.find_nearest(Vector(q))
                if loc is None:
                    continue
                nn = np.array(nn)
                sd = (q - np.array(loc)) @ nn
                if sd < h + 0.004:
                    q = np.array(loc) + nn * (h + 0.004)
                    nb = nn
                elif nb is None and dist < 0.05:
                    nb = nn
            if nb is None:
                nb = np.array([0, 1.0, 0.25]) / np.linalg.norm([0, 1.0, 0.25])
            pts.append(p), nrms.append(nb), phase.append(1.0)
            d_prev = (q - p) / max(np.linalg.norm(q - p), 1e-6)
            p = q
            if p[2] < z_end:
                break
    pts.append(p), nrms.append(nrms[-1]), phase.append(phase[-1])
    return np.array(pts), np.array(nrms), np.array(phase)


def _resample(P, N, Ph, n):
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    t = np.linspace(0, s[-1], n)
    f = lambda A: np.stack([np.interp(t, s, A[:, k]) for k in range(A.shape[1])], 1)
    N2 = f(N)
    N2 /= np.maximum(np.linalg.norm(N2, axis=1, keepdims=True), 1e-6)
    return f(P), N2, np.interp(t, s, Ph), t


def build_cards(body, collide, seed=21):
    rng = np.random.RandomState(seed)
    head_bvh = geo.bvh_from_object(body)
    coll = [geo.bvh_from_object(o) for o in collide]
    layers = [  # offset, width, count, min spacing
        (0.0024, 0.020, 150, 0.0095),
        (0.0046, 0.017, 115, 0.0110),
        (0.0072, 0.014, 70, 0.0140),
    ]
    verts, faces, uvs, info = [], [], [], []
    strands = []
    jobs = [(li, h, w0, r, None) for li, (h, w0, count, md) in enumerate(layers)
            for r in _sample_roots(body, count, md, rng)]
    # short fringe cards rooted on the hairline soften the scalp-cap edge
    jobs += [(3, 0.0016, 0.012, r, rng.uniform(0.04, 0.07))
             for r in _sample_roots(body, 90, 0.0075, rng, hairline=True)]
    for li, h, w0, r, max_len in jobs:
        zc = 1.29 + 0.07 * min(abs(r[0]) / 0.08, 1) ** 2 + rng.normal(0, 0.018)
        P, N, Ph = trace(r, h + rng.uniform(-0.0006, 0.0006), zc, head_bvh, coll, rng, max_len=max_len)
        if len(P) < 4:
            continue
        P, N, Ph, S = _resample(P, N, Ph, 20 if max_len is None else 6)
        if max_len is None:
            strands.append((P, Ph, S))
        col = rng.randint(N_HAIR_COLS)
        w = w0 * rng.uniform(0.8, 1.2)
        base = len(verts)
        L = S[-1]
        vs = 1.0 if max_len is None else min(1.0, L / 0.45)   # keep strand scale on short cards
        for k in range(len(P)):
            t = P[min(k + 1, len(P) - 1)] - P[max(k - 1, 0)]
            t /= np.linalg.norm(t)
            b = np.cross(N[k], t)
            b /= max(np.linalg.norm(b), 1e-6)
            tt = S[k] / L
            ww = w * (1 - 0.55 * tt ** 1.5) * (0.6 + 0.4 * smoothstep(0, 0.06, tt))
            lift = N[k] * 0.0012 * (1 - tt * 0.5)
            for j, u in enumerate((-0.5, 0.0, 0.5)):
                verts.append(P[k] + b * u * ww + (lift if j == 1 else 0))
                info.append((Ph[k], S[k], P[k][0], li))
        for k in range(len(P) - 1):
            for j in range(2):
                a = base + k * 3 + j
                faces.append((a, a + 1, a + 4, a + 3))
                v0, v1 = S[k] / L * vs, S[k + 1] / L * vs
                u0, u1 = (col + 0.08 + 0.42 * j) / N_COLS, (col + 0.08 + 0.42 * (j + 1)) / N_COLS
                uvs.append([(u0, 1 - v0), (u1, 1 - v0), (u1, 1 - v1), (u0, 1 - v1)])
    ob = mesh_object("HairCards", np.array(verts), faces, uvs)
    return ob, np.array(info), strands


def flyaways(body):
    """Two loose strands breaking from the temples - a little life in the slick-back."""
    head_bvh = geo.bvh_from_object(body)
    verts, faces, uvs = [], [], []
    for sx in (1, -1):
        root = np.array([0.045 * sx, -0.135, 1.900])
        loc, n, _, _ = head_bvh.find_nearest(Vector(root))
        p0 = np.array(loc) + np.array(n) * 0.008
        pts = []
        for k in range(18):
            t = k / 17
            pts.append(p0 + np.array([0.02 * sx * t + 0.006 * sx * math.sin(t * 6), -0.02 * math.sin(t * 2.2), -0.09 * t ** 1.3]))
        pts = np.array(pts)
        for k in range(len(pts)):            # keep off the face
            loc, n, _, _ = head_bvh.find_nearest(Vector(pts[k]))
            d = pts[k] - np.array(loc)
            if d @ np.array(n) < 0.007:
                pts[k] = np.array(loc) + np.array(n) * 0.007
        base = len(verts)
        for k, p in enumerate(pts):
            t = pts[min(k + 1, len(pts) - 1)] - pts[max(k - 1, 0)]
            t /= np.linalg.norm(t)
            b = np.cross(t, [0, -1.0, 0])
            b /= np.linalg.norm(b)
            w = 0.003 * (1 - 0.6 * k / 17)
            verts += [p - b * w, p + b * w]
        for k in range(len(pts) - 1):
            a = base + 2 * k
            faces.append((a, a + 1, a + 3, a + 2))
            v0, v1 = k / 17, (k + 1) / 17
            uvs.append([(3.2 / N_COLS, 1 - v0), (3.8 / N_COLS, 1 - v0), (3.8 / N_COLS, 1 - v1), (3.2 / N_COLS, 1 - v1)])
    return mesh_object("HairFlyaways", np.array(verts), faces, uvs)


# --------------------------------------------------------------------------------------
# hair bones

CHAINS = (("hair_l", 0.045), ("hair_c", 0.0), ("hair_r", -0.045))
N_HAIR = 4


def add_hair_bones(arm, strands):
    """Three 4-bone chains following the hanging hair (for engine-side physics)."""
    chains = {}
    for name, xc in CHAINS:
        pts = []
        for P, Ph, S in strands:
            hang = P[Ph > 0.5]
            if len(hang) < 4 or abs(hang[0][0] - xc) > 0.03:
                continue
            seg = np.linalg.norm(np.diff(hang, axis=0), axis=1)
            s = np.concatenate([[0], np.cumsum(seg)])
            t = np.linspace(0, s[-1], 20)
            pts.append(np.stack([np.interp(t / s[-1], s / s[-1], hang[:, k]) for k in range(3)], 1))
        avg = np.mean(pts, axis=0)
        idx = np.linspace(0, len(avg) - 1, N_HAIR + 1).round().astype(int)
        chains[name] = avg[idx]
    set_active(arm)
    bpy.ops.object.mode_set(mode="EDIT")
    eb = arm.data.edit_bones
    for name, J in chains.items():
        parent = eb["head"]
        for k in range(N_HAIR):
            b = eb.new(f"{name}_{k + 1:02d}")
            b.head, b.tail = Vector(J[k]), Vector(J[k + 1])
            b.parent = parent
            b.use_connect = k > 0
            y = (b.tail - b.head).normalized()
            x = Vector((1, 0, 0))
            x = (x - x.dot(y) * y).normalized()
            b.align_roll(x.cross(y))
            parent = b
    bpy.ops.object.mode_set(mode="OBJECT")
    return chains


def hair_weights(co, chains, names, phase_blend):
    """Dense weights: head on the scalp, blended chain bones down the length.

    phase_blend: 0 = scalp (head), 1 = fully on the chains."""
    W = np.zeros((len(co), len(names)), dtype=np.float32)
    hi = names.index("head")
    xw = []
    for name, xc in CHAINS:
        J = chains[name]
        s, c, d, seg = geo.chain_param(co, list(J))
        L = np.linalg.norm(np.diff(J, axis=0), axis=1)
        cum = np.concatenate([[0], np.cumsum(L)])
        pos = np.interp(s, cum, np.arange(len(J)))
        bw = np.zeros((len(co), N_HAIR))
        for k in range(N_HAIR):
            bw[:, k] = np.clip(1 - np.abs(pos - (k + 0.5)), 0, 1)
        bw[pos < 0.5, 0] = 1
        bw[pos > N_HAIR - 0.5, N_HAIR - 1] = 1
        bw /= np.maximum(bw.sum(1, keepdims=True), 1e-6)
        g = np.exp(-((co[:, 0] - xc) / 0.035) ** 2)
        xw.append((name, bw, g))
    gsum = sum(g for _, _, g in xw) + 1e-6
    for name, bw, g in xw:
        for k in range(N_HAIR):
            W[:, names.index(f"{name}_{k + 1:02d}")] += (bw[:, k] * g / gsum * phase_blend).astype(np.float32)
    W[:, hi] += (1 - phase_blend).astype(np.float32)
    return W


# --------------------------------------------------------------------------------------
# brows and lashes

def brows(body):
    bvh = geo.bvh_from_object(body)
    verts, faces, uvs = [], [], []
    for sx in (1, -1):
        for layer, (dz, lift, wmul) in enumerate(((0.0, 0.0007, 1.0), (0.001, 0.0017, 0.7))):
            pts, nrm = [], []
            for t in np.linspace(0, 1, 14):
                x = (0.011 + 0.054 * t) * sx
                z = 1.8485 + 0.0065 * t + 0.0035 * math.sin(math.pi * t) + dz
                hit = bvh.ray_cast(Vector((x, -0.5, z)), Vector((0, 1, 0)))
                if hit[0] is None:
                    continue
                pts.append(np.array(hit[0]) + np.array(hit[1]) * lift)
                nrm.append(np.array(hit[1]))
            base = len(verts)
            n = len(pts)
            for k in range(n):
                t = pts[min(k + 1, n - 1)] - pts[max(k - 1, 0)]
                t /= np.linalg.norm(t)
                b = np.cross(nrm[k], t)
                b /= np.linalg.norm(b)
                if b[2] < 0:
                    b = -b
                tt = k / (n - 1)
                w = (0.0075 - 0.0045 * tt ** 1.2) * wmul
                verts += [pts[k] - b * w * 0.5, pts[k] + b * w * 0.5]
            for k in range(n - 1):
                a = base + 2 * k
                faces.append((a, a + 1, a + 3, a + 2) if sx > 0 else (a, a + 2, a + 3, a + 1))
                v0, v1 = 0.15 + 0.55 * k / (n - 1), 0.15 + 0.55 * (k + 1) / (n - 1)
                c0, c1 = (LASH_COL + 0.15) / N_COLS, (LASH_COL + 0.85) / N_COLS
                if sx > 0:
                    uvs.append([(c0, 1 - v0), (c1, 1 - v0), (c1, 1 - v1), (c0, 1 - v1)])
                else:
                    uvs.append([(c0, 1 - v0), (c0, 1 - v1), (c1, 1 - v1), (c1, 1 - v0)])
    return mesh_object("Brows", np.array(verts), faces, uvs)


def lashes(base, v_mh):
    """MakeHuman eyelash helper strips, re-UV'd onto the lash column of the hair atlas."""
    from .body import submesh
    groups = base.groups_matching("helper-l-eyelashes") + base.groups_matching("helper-r-eyelashes")
    vb, f, _, used = submesh(base, v_mh, groups)
    uvs = []
    for face in f:
        p = vb[list(face)]
        uv = []
        for q in p:
            sx = 1 if q[0] > 0 else -1
            c = np.array([0.034 * sx, -0.128, 1.830])
            rel = q - c
            u = np.clip((abs(q[0]) - 0.018) / 0.036, 0, 1)
            vv = np.clip((np.linalg.norm(rel) - 0.016) / 0.012, 0, 1)
            uv.append(((LASH_COL + 0.1 + 0.8 * u) / N_COLS, 1 - (0.05 + 0.9 * vv)))
        uvs.append(uv)
    ob = mesh_object("Lashes", vb, f, uvs)
    return ob, used
