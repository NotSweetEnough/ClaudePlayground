"""Shaolin-style garments built from MakeHuman's fitting helpers (tights / skirt).

Every garment starts as a helper shell (so it inherits MakeHuman skin weights by
vertex id), is subdivided, then loosened with radial ease + pseudo-cloth draping
over the body, and finally gets procedural fold displacement.
"""
import math

import bpy
import numpy as np

from . import folds, geo, mh
from .butil import get_verts, mesh_object, set_active, set_verts, smoothstep


class Ctx:
    """Shared build context."""

    def __init__(self, base, v_mh, body_ob, arm_ob, bone_names, Wfull):
        self.base, self.v_mh = base, v_mh
        self.J = {}
        for b in arm_ob.data.bones:
            self.J[b.name + ".head"] = np.array(b.head_local)
            self.J[b.name + ".tail"] = np.array(b.tail_local)
        self.body, self.arm = body_ob, arm_ob
        self.bone_names, self.Wfull = bone_names, Wfull
        self.body_bvh = geo.bvh_from_object(body_ob)

    def j(self, name):
        return np.asarray(self.J[name], dtype=float)


# --------------------------------------------------------------------------------------
# generic piece handling

def extract(ctx, name, groups, keep):
    """Create an object from helper faces where keep(face_centers, face_vertex_coords) is True."""
    base = ctx.base
    fidx = np.concatenate([base.group_faces(g) for g in groups])
    vb = mh.to_blender(ctx.v_mh)
    fc = vb[base.faces[fidx]].mean(1)
    m = keep(fc, vb[base.faces[fidx]])
    fidx = fidx[m]
    faces = base.faces[fidx]
    used = np.unique(faces.ravel())
    remap = -np.ones(len(base.verts), dtype=np.int64)
    remap[used] = np.arange(len(used))
    ob = mesh_object(name, vb[used], remap[faces], None)
    W = limit(ctx.Wfull[used])
    write_groups(ob, W, ctx.bone_names)
    return ob


def limit(W, k=4):
    from .rig import limit_normalize
    return limit_normalize(W, k)


def write_groups(ob, W, names):
    ob.vertex_groups.clear()
    for j, b in enumerate(names):
        nz = np.nonzero(W[:, j] > 0)[0]
        if len(nz) == 0:
            continue
        vg = ob.vertex_groups.new(name=b)
        for i in nz:
            vg.add([int(i)], float(W[i, j]), "REPLACE")


def subdivide(ob, levels=1):
    m = ob.modifiers.new("sub", "SUBSURF")
    m.levels = m.render_levels = levels
    m.boundary_smooth = "PRESERVE_CORNERS"
    set_active(ob)
    bpy.ops.object.modifier_apply(modifier=m.name)


def faces_of(ob):
    return [tuple(p.vertices) for p in ob.data.polygons]


def bone_w(ob, names, prefixes):
    """Sum of vertex-group weights whose names start with any prefix."""
    n = len(ob.data.vertices)
    out = np.zeros(n)
    gi = [vg.index for vg in ob.vertex_groups if any(vg.name.startswith(p) for p in prefixes)]
    gi = set(gi)
    for v in ob.data.vertices:
        for g in v.groups:
            if g.group in gi:
                out[v.index] += g.weight
    return out


# --------------------------------------------------------------------------------------
# tunic

WAIST_Z = 1.10        # sash centre (soles lifted by body.SOLE)
SASH_H = 0.105


def arm_chain(ctx, s):
    return [ctx.j(f"upperarm_{s}.head"), ctx.j(f"upperarm_{s}.tail"), ctx.j(f"lowerarm_{s}.tail")]


def bisect(ob, co, no, clear_outer=False, clear_inner=False, vert_filter=None):
    """Cut `ob` with a plane; optionally restricted to vertices passing vert_filter(co_array)."""
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    bm.verts.ensure_lookup_table()
    if vert_filter is not None:
        pts = np.array([v.co for v in bm.verts])
        ok = vert_filter(pts)
        geom = [f for f in bm.faces if all(ok[v.index] for v in f.verts)]
        es = {e for f in geom for e in f.edges}
        vs = {v for f in geom for v in f.verts}
        geom = list(vs) + list(es) + geom
    else:
        geom = list(bm.verts) + list(bm.edges) + list(bm.faces)
    bmesh.ops.bisect_plane(bm, geom=geom, dist=1e-5, plane_co=tuple(co), plane_no=tuple(no),
                           clear_outer=clear_outer, clear_inner=clear_inner)
    bm.to_mesh(ob.data)
    bm.free()
    ob.data.update()


def delete_faces(ob, mask_fn):
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    fc = np.array([f.calc_center_median() for f in bm.faces])
    m = mask_fn(fc)
    bmesh.ops.delete(bm, geom=[f for f, d in zip(bm.faces, m) if d], context="FACES")
    loose = [v for v in bm.verts if not v.link_faces]
    bmesh.ops.delete(bm, geom=loose, context="VERTS")
    bm.to_mesh(ob.data)
    bm.free()
    ob.data.update()


# crossover collar (youren): outer (wearer-left) panel edge A, inner panel edge B, crossing at C
COLLAR_C = np.array([-0.022, 1.45])            # (x, z) of the crossing point
COLLAR_A_TOP = np.array([0.078, 1.735])
COLLAR_B_TOP = np.array([-0.075, 1.715])
COLLAR_A_END = np.array([-0.175, 1.155])        # A continues down to the wearer's right side
NECK_Z = 1.70


def _line_plane(p0, p1, above_point):
    d = np.array([p1[0] - p0[0], 0.0, p1[1] - p0[1]])
    n = np.cross(d, [0.0, 1.0, 0.0])
    t = np.array([above_point[0] - p0[0], 0.0, above_point[1] - p0[1]])
    if n @ t < 0:
        n = -n
    return np.array([p0[0], 0.0, p0[1]]), n / np.linalg.norm(n)


def in_vneck(p):
    ca, na = _line_plane(COLLAR_C, COLLAR_A_TOP, COLLAR_C + [0, 0.1])
    cb, nb = _line_plane(COLLAR_C, COLLAR_B_TOP, COLLAR_C + [0, 0.1])
    return ((p - ca) @ na > 0) & ((p - cb) @ nb > 0) & (p[:, 1] < 0.0) & (p[:, 2] > COLLAR_C[1] - 0.02)


def collar_distance(co):
    """Distance (xz) to the collar lines / neck cut, used to keep folds off the trims."""
    da = np.abs((co - _line_plane(COLLAR_C, COLLAR_A_TOP, COLLAR_C + [0, 0.1])[0]) @ _line_plane(COLLAR_C, COLLAR_A_TOP, COLLAR_C + [0, 0.1])[1])
    db = np.abs((co - _line_plane(COLLAR_C, COLLAR_B_TOP, COLLAR_C + [0, 0.1])[0]) @ _line_plane(COLLAR_C, COLLAR_B_TOP, COLLAR_C + [0, 0.1])[1])
    de = np.abs((co - _line_plane(COLLAR_C, COLLAR_A_END, COLLAR_C + [-0.1, 0.1])[0]) @ _line_plane(COLLAR_C, COLLAR_A_END, COLLAR_C + [-0.1, 0.1])[1])
    front = co[:, 1] < 0.02
    upper = co[:, 2] > COLLAR_C[1] - 0.03
    d = np.where(front & upper, np.minimum(da, db), 1.0)
    d = np.minimum(d, np.where(front & ~upper & (co[:, 2] > WAIST_Z), de, 1.0))
    d = np.minimum(d, np.abs(co[:, 2] - NECK_Z) + 0.0 * co[:, 0])
    return d


def tunic_folds(ctx, co, F, armw, sleeve_len, seed=11):
    disp = np.zeros(len(co))
    for k, (s, sx) in enumerate((("l", 1), ("r", -1))):
        ch = arm_chain(ctx, s)
        L = np.linalg.norm(ch[1] - ch[0]) + sleeve_len
        sv, phi, rad, rn = folds.limb_coords(co, ch)
        w = armw * (co[:, 0] * sx > 0)
        t = sv / L
        long = folds.pipes(phi * 0.09, sv, 0.055, 0.22, seed + k)
        amp_long = 0.011 * smoothstep(0.05, 0.75, t) * (0.5 + 0.5 * np.clip(-rn[:, 2], 0, 1))
        ring = folds.pipes(sv, phi * 0.09, 0.045, 0.12, seed + 5 + k, 0.8)
        amp_ring = 0.008 * smoothstep(0.55, 0.95, t)
        pit = folds.pipes(phi * 0.09 + sv * 0.6, sv, 0.04, 0.08, seed + 9 + k)
        amp_pit = 0.006 * smoothstep(0.3, 0.0, t) * np.clip(-rn[:, 2] * 1.5, 0, 1)
        disp += w * (long * amp_long + ring * amp_ring + pit * amp_pit)
    yc = 0.5 * (co[:, 1].min() + co[:, 1].max())
    phi, rad = folds.torso_coords(co, yc)
    z = co[:, 2]
    tor = 1 - armw
    arc = phi * 0.17
    vert = folds.pipes(arc, z, 0.07, 0.30, seed + 20)
    amp_v = 0.002 + 0.009 * smoothstep(1.48, 1.2, z)
    blouse = folds.pipes(z, arc, 0.035, 0.09, seed + 21, 1.0)
    amp_b = 0.010 * smoothstep(WAIST_Z + 0.17, WAIST_Z + 0.07, z)
    sgn = np.sign(co[:, 0])
    diag = folds.pipes(arc * sgn * 0.8 + z * 0.6, z - arc * sgn * 0.8, 0.06, 0.16, seed + 22)
    amp_d = 0.007 * smoothstep(1.5, 1.35, z) * smoothstep(1.12, 1.25, z) * smoothstep(0.06, 0.16, np.abs(co[:, 0]))
    disp += tor * (vert * amp_v + blouse * amp_b + diag * amp_d)
    disp *= smoothstep(0.004, 0.04, collar_distance(co))
    return folds.apply(co, F, disp, ctx.body_bvh, 0.004)


def sleeve_end(ctx, s, sleeve_len):
    ch = arm_chain(ctx, s)
    fdir = (ch[2] - ch[1]) / np.linalg.norm(ch[2] - ch[1])
    udir = (ch[1] - ch[0]) / np.linalg.norm(ch[1] - ch[0])
    nrm = udir * 0.35 + fdir * 0.65            # cut plane tilted between upper/fore arm
    nrm /= np.linalg.norm(nrm)
    return ch[1] + fdir * sleeve_len, nrm


def tunic_upper(ctx, sleeve_len=0.04):
    """Torso + wide sleeves, from just under the sash up to the neck."""
    cut_z = WAIST_Z - SASH_H * 0.35

    def keep(fc, fv):
        k = fc[:, 2] > cut_z - 0.05
        for s, sx in (("l", 1), ("r", -1)):
            ch = arm_chain(ctx, s)
            L_up = np.linalg.norm(ch[1] - ch[0])
            sp, _, d, _ = geo.chain_param(fc, ch)
            onarm = (fc[:, 0] * sx > abs(ch[0][0]) * 0.9)
            k &= ~(onarm & (sp > L_up + sleeve_len + 0.05))
        return k

    ob = extract(ctx, "Tunic", ["helper-tights"], keep)
    subdivide(ob, 2)

    co = get_verts(ob)
    F = faces_of(ob)
    A = geo.adjacency(F, len(co))
    armw = np.clip(bone_w(ob, ctx.bone_names, ["upperarm", "lowerarm", "hand"]) * 1.4, 0, 1)
    edge = geo.boundary_mask(F, len(co))

    # --- torso ease: snug at shoulders, blouses above the sash ----------------------
    z = co[:, 2]
    torso_off = 0.010 + 0.022 * smoothstep(1.50, 1.20, z) - 0.014 * smoothstep(WAIST_Z + 0.08, WAIST_Z, z)
    off = torso_off * (1 - armw) + 0.006 * armw
    co = geo.drape(co, A, ctx.body_bvh, off, iters=12, lam=0.55, pin=edge)

    # --- sleeves: bell-shaped radial ease around the arm axis ----------------------
    for s, sx in (("l", 1), ("r", -1)):
        ch = arm_chain(ctx, s)
        L_up = np.linalg.norm(ch[1] - ch[0])
        sp, c, d, _ = geo.chain_param(co, ch)
        side = (co[:, 0] * sx > 0)
        t = np.clip(sp / (L_up + sleeve_len), 0, 1)
        ease = 0.012 + 0.05 * smoothstep(0.1, 1.0, t) ** 1.2
        r = co - c
        rn = r / np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-6)
        w = (armw * side)[:, None]
        co = co + rn * ease[:, None] * w
        sag = 0.025 * smoothstep(0.2, 1.0, t) * np.clip(-rn[:, 2], 0, 1)
        co[:, 2] -= sag * w[:, 0]

    co = geo.smooth(co, A, 0.4, 2, mask=(~edge).astype(float))
    co = tunic_folds(ctx, co, F, armw, sleeve_len)
    set_verts(ob, co)

    # clean cuts after shaping: hem under the sash, sleeve ends, crossover V neck, back neck
    bisect(ob, (0, 0, cut_z), (0, 0, -1), clear_outer=True)
    for s, sx in (("l", 1), ("r", -1)):
        pc, pn = sleeve_end(ctx, s, sleeve_len)
        bisect(ob, pc, pn, clear_outer=True, vert_filter=lambda p, sx=sx: p[:, 0] * sx > 0.2)
    for top in (COLLAR_A_TOP, COLLAR_B_TOP):
        c, n = _line_plane(COLLAR_C, top, COLLAR_C + [0, 0.1])
        bisect(ob, c, n, vert_filter=lambda p: (p[:, 1] < 0.02) & (p[:, 2] > 1.40) & (np.abs(p[:, 0]) < 0.16))
    delete_faces(ob, in_vneck)
    bisect(ob, (0, 0, NECK_Z), (0, 0, 1), clear_outer=True, vert_filter=lambda p: p[:, 2] > 1.58)
    return ob


# --------------------------------------------------------------------------------------
# loops, weights, trims

def boundary_loops(ob):
    """Ordered boundary vertex loops of a mesh."""
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    nxt = {}
    for e in bm.edges:
        if len(e.link_faces) == 1:
            a, b = e.verts[0].index, e.verts[1].index
            nxt.setdefault(a, []).append(b)
            nxt.setdefault(b, []).append(a)
    bm.free()
    seen, loops = set(), []
    for start in nxt:
        if start in seen:
            continue
        loop, prev, cur = [start], None, start
        seen.add(start)
        while True:
            cand = [x for x in nxt[cur] if x != prev and x not in seen]
            if not cand:
                break
            prev, cur = cur, cand[0]
            loop.append(cur)
            seen.add(cur)
        loops.append(loop)
    return loops


def weights_from(src_ob, dst_co, names, k=6):
    """Inverse-distance blended weights from the k nearest vertices of src_ob (dense matrix)."""
    from scipy.spatial import cKDTree
    from .rig import read_weights
    src_co = get_verts(src_ob)
    Ws = read_weights(src_ob, names)
    d, i = cKDTree(src_co).query(dst_co, k=k)
    w = 1.0 / np.maximum(d, 1e-5) ** 2
    w /= w.sum(1, keepdims=True)
    W = (Ws[i] * w[:, :, None]).sum(1)
    return limit(W)


def surface_frame(bvh, p):
    from mathutils import Vector
    loc, nrm, _, _ = bvh.find_nearest(Vector(p))
    return np.array(loc), np.array(nrm)


def loft(name, rings, closed=True, cap=False):
    """Mesh from a list of rings (each (M,3)); rings run along the first axis."""
    rings = np.asarray(rings)
    R, M = rings.shape[:2]
    verts = rings.reshape(-1, 3)
    faces, uvs = [], []
    span = M if closed else M - 1
    for r in range(R - 1):
        for m in range(span):
            a, b = r * M + m, r * M + (m + 1) % M
            faces.append((a, b, b + M, a + M))
            u0, u1 = m / span, (m + 1) / span
            v0, v1 = r / (R - 1), (r + 1) / (R - 1)
            uvs.append([(u0, v0), (u1, v0), (u1, v1), (u0, v1)])
    return mesh_object(name, verts, faces, uvs)


def ribbon_profile(w, h):
    """Cross-section of a raised fabric band: (across, height) pairs."""
    return [(-0.003, -0.002), (-0.001, h * 0.7), (0.004, h), (w * 0.5, h * 1.08),
            (w - 0.004, h), (w + 0.001, h * 0.6), (w + 0.003, -0.001)]


def collar_band(ctx, tunic, width=0.036, height=0.0045):
    from mathutils import Vector
    bvh = geo.bvh_from_object(tunic)
    co = get_verts(tunic)
    loops = boundary_loops(tunic)
    neck = max(loops, key=lambda l: co[l, 2].mean() + (co[l, 1] < 0).mean())
    L = co[neck]
    i0 = int(np.argmin(np.linalg.norm(L[:, [0, 2]] - COLLAR_C, axis=1)))
    L = np.roll(L, -i0, axis=0)
    if L[3, 0] < L[0, 0]:             # walk up along A (towards +x) first
        L = np.roll(L[::-1], 1, axis=0)
    # A continuation below the crossing: raycast onto the tunic front
    ext = []
    for t in np.linspace(1.0, 0.0, 14)[:-1]:
        xz = COLLAR_C + (COLLAR_A_END - COLLAR_C) * t
        hit = bvh.ray_cast(Vector((xz[0], -0.6, xz[1])), Vector((0, 1, 0)))
        if hit[0] is not None:
            ext.append(np.array(hit[0]))
    path = np.vstack([ext, L])
    # resample evenly
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    n = max(int(s[-1] / 0.006), 20)
    ss = np.linspace(0, s[-1], n)
    path = np.stack([np.interp(ss, s, path[:, k]) for k in range(3)], 1)
    prof = ribbon_profile(width, height)
    rings, prev_b = [], None
    n_ext = np.searchsorted(ss, s[len(ext)])
    for i, p in enumerate(path):
        t = path[min(i + 1, n - 1)] - path[max(i - 1, 0)]
        t /= np.linalg.norm(t)
        _, nr = surface_frame(bvh, p)
        b = np.cross(nr, t)
        b /= np.linalg.norm(b)
        if i >= n_ext:
            # choose the side that stays on cloth (the other side is the neck opening)
            q = np.array([p + b * 0.012, p - b * 0.012])
            hole = in_vneck(q) | (q[:, 2] > NECK_Z - 0.004)
            if hole[0] and not hole[1]:
                b = -b
            elif hole[0] == hole[1] and prev_b is not None and b @ prev_b < 0:
                b = -b
        else:
            if prev_b is None and b[0] - b[2] < 0:   # outer panel side: towards +x / down
                b = -b
            elif prev_b is not None and b @ prev_b < 0:
                b = -b
        prev_b = b
        tuck = smoothstep(n - 1, n - 8, i) if i >= n_ext else 1.0
        ring = []
        for u, h in prof:
            h = h * (0.35 + 0.65 * tuck) - 0.002 * (1 - tuck)
            q, nq = surface_frame(bvh, p + b * max(u, 0))
            if u < 0:
                q = p + b * u
            ring.append(q + nq * h)
        rings.append(ring)
    band = loft("CollarBand", np.array(rings), closed=False)
    # recess the inner panel along the overlap edge so the wrap reads
    pts = co.copy()
    ca, na = _line_plane(COLLAR_C, COLLAR_A_END, COLLAR_C + [-0.1, 0.1])
    dist = (pts - ca) @ na
    m = (dist > 0) & (dist < 0.05) & (pts[:, 1] < 0) & (pts[:, 2] < COLLAR_C[1] + 0.01) & (pts[:, 2] > WAIST_Z)
    nrm = geo.vertex_normals(pts, faces_of(tunic))
    pts[m] -= nrm[m] * (0.004 * (1 - dist[m] / 0.05))[:, None]
    set_verts(tunic, pts)
    return band


def sleeve_cuffs(ctx, tunic, sleeve_len=0.04, seed=3):
    co = get_verts(tunic)
    out = []
    for s, sx in (("l", 1), ("r", -1)):
        pc, pn = sleeve_end(ctx, s, sleeve_len)
        loops = [l for l in boundary_loops(tunic) if np.mean(co[l, 0]) * sx > 0.25]
        loop = max(loops, key=len)
        P = co[loop]
        P = P[np.argsort(np.arctan2(*(_basis_coords(P - P.mean(0), pn))))]
        ctr = P.mean(0)
        rng = np.random.RandomState(seed + (s == "l"))
        M = len(P)
        k = np.arange(M) / M * 2 * np.pi
        thick = 1 + 0.25 * np.sin(k * 3 + rng.rand() * 6) + 0.15 * np.sin(k * 7 + rng.rand() * 6)
        length = 1 + 0.2 * np.sin(k * 2 + rng.rand() * 6) + 0.1 * np.sin(k * 5 + rng.rand() * 6)
        prof = [(-0.004, 0.0), (0.001, 0.010), (0.010, 0.016), (0.019, 0.010), (0.023, -0.010),
                (0.022, -0.035), (0.019, -0.055), (0.012, -0.066), (0.004, -0.068)]
        rings = []
        for rr, aa in prof:
            ring = []
            for j, p in enumerate(P):
                rad = p - ctr
                rad -= (rad @ pn) * pn
                rad /= np.linalg.norm(rad)
                ring.append(p + rad * rr * thick[j] + pn * aa * (length[j] if aa < 0 else 1.0))
            rings.append(ring)
        out.append(loft(f"Cuff_{s}", np.array(rings), closed=True))
    return out


def _basis_coords(v, n):
    a = np.cross(n, [0, 0, 1.0])
    if np.linalg.norm(a) < 1e-3:
        a = np.cross(n, [0, 1.0, 0])
    a /= np.linalg.norm(a)
    b = np.cross(n, a)
    return v @ a, v @ b


def attach(ctx, parts, target):
    """Give each part weights from `target` and join them into it."""
    for p in parts:
        W = weights_from(target, get_verts(p), ctx.bone_names)
        write_groups(p, W, ctx.bone_names)
    set_active(target)
    for p in parts:
        p.select_set(True)
    bpy.context.view_layer.objects.active = target
    bpy.ops.object.join()
    return target


# --------------------------------------------------------------------------------------
# tunic skirt (lower half, under the sash), sash, pants, leg wraps

HEM_Z = 0.69          # tunic hem, mid thigh
SLIT_TOP = 0.90       # side slits run from the hem up to here
WRAP_TOP = 0.47       # leg wraps: from the shoe collar up to just under the knee
WRAP_BOT = 0.085


def _center_y(co):
    return 0.5 * (co[:, 1].min() + co[:, 1].max())


def split_seam(ob, sx, yc, z_top, x_min=0.1):
    """Rip the mesh along the side seam (plane y=yc) below z_top. Returns (front_ids, back_ids)."""
    import bmesh
    bisect(ob, (0, yc, 0), (0, 1, 0), vert_filter=lambda p: (p[:, 0] * sx > x_min) & (p[:, 2] < z_top))
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    edges = [e for e in bm.edges
             if all(abs(v.co.y - yc) < 1e-4 and v.co.x * sx > x_min and v.co.z < z_top - 0.005 for v in e.verts)]
    bmesh.ops.split_edges(bm, edges=edges)
    bm.verts.ensure_lookup_table()
    front, back = [], []
    for v in bm.verts:
        if v.co.x * sx > x_min * 0.5 and v.co.z < z_top + 0.05 and v.link_faces:
            fy = np.mean([f.calc_center_median().y for f in v.link_faces])
            (front if fy < yc else back).append(v.index)
    bm.to_mesh(ob.data)
    bm.free()
    return np.array(front, dtype=int), np.array(back, dtype=int)


def tunic_lower(ctx):
    top = WAIST_Z + SASH_H * 0.3
    ob = extract(ctx, "TunicSkirt", ["helper-skirt"],
                 lambda fc, fv: (fc[:, 2] < top + 0.06) & (fc[:, 2] > HEM_Z - 0.08))
    subdivide(ob, 2)
    co = get_verts(ob)
    A = geo.adjacency(faces_of(ob), len(co))
    yc = _center_y(co[co[:, 2] < 0.9])
    z = co[:, 2]
    r = co[:, :2] - [0.0, yc]
    rl = np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-6)
    flare = 0.004 + 0.045 * smoothstep(top - 0.05, HEM_Z, z) ** 1.3
    co[:, :2] += r / rl * flare[:, None]
    co = geo.drape(co, A, ctx.body_bvh, 0.016, iters=6, lam=0.45)
    phi, rad = folds.torso_coords(co, yc)
    arc = phi * 0.2
    z = co[:, 2]
    vert = folds.pipes(arc, z, 0.085, 0.4, 61)
    diag = folds.pipes(arc + z * 0.4, z, 0.06, 0.14, 62)
    disp = vert * (0.004 + 0.016 * smoothstep(top, HEM_Z, z)) + diag * 0.004 * smoothstep(top - 0.05, HEM_Z + 0.1, z)
    disp *= smoothstep(top - 0.01, top - 0.06, z)
    co = folds.apply(co, faces_of(ob), disp, ctx.body_bvh, 0.006)
    set_verts(ob, co)
    bisect(ob, (0, 0, HEM_Z), (0, 0, -1), clear_outer=True)
    bisect(ob, (0, 0, top), (0, 0, 1), clear_outer=True)
    for sx in (1, -1):
        front, back = split_seam(ob, sx, yc, SLIT_TOP, 0.12)
        co = get_verts(ob)
        g = 0.014 * smoothstep(SLIT_TOP, HEM_Z, co[:, 2])
        near = lambda ids: ids[np.abs(co[ids, 1] - yc) < 0.08]
        f, b = near(front), near(back)
        w = lambda ids: g[ids] * np.clip(1 - np.abs(co[ids, 1] - yc) / 0.08, 0, 1)
        co[f, 1] -= w(f)
        co[b, 1] += w(b)
        co[f, 0] -= sx * w(f) * 0.3
        co[b, 0] -= sx * w(b) * 0.3
        set_verts(ob, co)
    return ob


def outer_radius(bvhs, center, dirs, z, start=0.30):
    """Outermost surface distance along horizontal directions from a vertical axis."""
    from mathutils import Vector
    out = np.zeros(len(dirs))
    for i, d in enumerate(dirs):
        o = Vector((center[0] + d[0] * start, center[1] + d[1] * start, z))
        best = 0.0
        for bvh in bvhs:
            hit = bvh.ray_cast(o, Vector((-d[0], -d[1], 0.0)))
            if hit[0] is not None:
                best = max(best, start - hit[3])
        out[i] = best
    return out


def sash(ctx, garments, avoid=(), n_ang=128, n_z=11, seed=7):
    bvhs = [geo.bvh_from_object(g) for g in garments]
    co_t = get_verts(garments[0])
    band = co_t[np.abs(co_t[:, 2] - WAIST_Z) < 0.05]
    yc = _center_y(band)
    ang = np.linspace(0, 2 * np.pi, n_ang, endpoint=False)
    dirs = np.stack([np.cos(ang), np.sin(ang)], 1)
    zs = np.linspace(WAIST_Z - SASH_H / 2, WAIST_Z + SASH_H / 2, n_z)
    rmax = np.zeros(n_ang)
    for zz in np.linspace(zs[0] - 0.01, zs[-1] + 0.01, 7):
        rmax = np.maximum(rmax, outer_radius(bvhs, (0, yc), dirs, zz))
    rmax = np.maximum(rmax, 0.05)
    for _ in range(10):
        rmax = np.maximum(rmax, 0.5 * rmax + 0.25 * (np.roll(rmax, 1) + np.roll(rmax, -1)))
    rings = []
    for k, zz in enumerate(zs):
        t = (zz - zs[0]) / (zs[-1] - zs[0])
        edge = 1 - (2 * t - 1) ** 14                     # crisp but rounded top / bottom edges
        layer = 0.0012 * (np.abs(np.sin(t * np.pi * 3)) > 0.85)  # wrapped layers step
        bulge = 0.005 + 0.0015 * np.sin(t * np.pi) + layer
        wav = 0.0010 * np.sin(ang * 9 + t * 5 + 1.3) + 0.0006 * np.sin(ang * 23 + 2.0)
        rr = rmax + bulge * edge - 0.003 * (1 - edge) + wav * edge
        rings.append(np.stack([dirs[:, 0] * rr, yc + dirs[:, 1] * rr, np.full(n_ang, zz)], 1))
    rings = np.array(rings)
    belt = loft("Sash", rings, closed=True)
    # close the band at top/bottom with an inward lip
    knot, tails = sash_knot(ctx, rings, yc, seed, avoid)
    return belt, knot, tails


def sash_knot(ctx, rings, yc, seed, avoid=()):
    """Square knot on the wearer's left hip with two hanging tails."""
    import bmesh
    ang_k = math.radians(-38)                   # front-left (character faces -y, left = +x)
    d = np.array([math.cos(ang_k), math.sin(ang_k)])
    mid = rings[len(rings) // 2]
    i = int(np.argmin(np.linalg.norm((mid[:, :2] - [0, yc]) / np.linalg.norm(mid[:, :2] - [0, yc], axis=1, keepdims=True) - d, axis=1)))
    p = mid[i].copy()
    nrm = np.array([d[0], d[1], 0.0])
    up = np.array([0, 0, 1.0])
    side = np.cross(up, nrm)
    diag = up * math.cos(math.radians(28)) + side * math.sin(math.radians(28))
    # knot volume: a flattened cushion
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=24, v_segments=14, radius=1.0)
    me = bpy.data.meshes.new("SashKnot")
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new("SashKnot", me)
    bpy.context.scene.collection.objects.link(ob)
    co = get_verts(ob)
    x, y, z = co[:, 0], co[:, 1], co[:, 2]
    sq = lambda a, e: np.sign(a) * np.abs(a) ** e         # superellipsoid-ish: boxier cushion
    local = np.stack([sq(x, 0.7) * 0.040, sq(y, 0.8) * 0.016, sq(z, 0.7) * 0.034], 1)
    w = local[:, 0, None] * side + (local[:, 1, None] + 0.012) * nrm + local[:, 2, None] * up
    set_verts(ob, p + w)
    me.shade_smooth()
    # wrapping loop over the cushion (the tied band), diagonal
    loop_rings = []
    for k in range(48):
        a = 2 * np.pi * k / 48
        c = p + nrm * (0.010 + 0.024 * math.cos(a)) + diag * (0.042 * math.sin(a))
        loop_rings.append([c + np.cross(diag, nrm) * u * 0.05 + nrm * (0.002 * (1 - (2 * u) ** 2))
                           for u in np.linspace(-0.5, 0.5, 6)])
    loop_ob = loft("SashKnotLoop", np.array(loop_rings).transpose(1, 0, 2), closed=True)
    loop_ob.data.shade_smooth()
    # tails
    tails = []
    for j, (L, off, wd) in enumerate(((0.44, -0.018, 0.075), (0.36, 0.022, 0.068))):
        n = 28
        pts, ws = [], []
        for k in range(n):
            t = k / (n - 1)
            sway = 0.02 * math.sin(t * 2.3 + j) + 0.012 * t * (1 if j else -1)
            q = p + side * (off + sway) + nrm * (0.018 + 0.03 * t) - up * (0.02 + L * t)
            pts.append(q)
            ws.append(wd * (1 - 0.15 * t))
        pts = np.array(pts)
        rings2 = []
        for k in range(n):
            t = k / (n - 1)
            tw = 0.35 * math.sin(t * 2.0 + j * 1.7)          # gentle twist
            ax = side * math.cos(tw) + nrm * math.sin(tw)
            fold = 0.006 * math.sin(t * 9 + j)
            row = []
            for u in np.linspace(-0.5, 0.5, 7):
                bow = (u * u - 0.25) * 0.012               # cupped cross-section
                row.append(pts[k] + ax * u * ws[k] + nrm * (bow + fold * u))
            rings2.append(row)
        rings2 = np.array(rings2)
        bv = [geo.bvh_from_object(g) for g in avoid]
        prev = 0.0
        for k in range(len(rings2)):                   # move each cross-section rigidly off the cloth
            need = prev * 0.8
            for b in bv:
                pushed = geo.push_out(rings2[k], b, 0.010)
                need = max(need, float(np.max(np.linalg.norm(pushed - rings2[k], axis=1))))
            rings2[k] += nrm * need
            prev = need
        tails.append(loft(f"SashTail{j}", rings2, closed=False))
    return [ob, loop_ob], tails


def leg_chain(ctx, s):
    return [ctx.j(f"thigh_{s}.head"), ctx.j(f"calf_{s}.head"), ctx.j(f"foot_{s}.head")]


def pants(ctx):
    top = WAIST_Z + 0.03
    bot = WRAP_TOP - 0.035
    ob = extract(ctx, "Pants", ["helper-tights"],
                 lambda fc, fv: (fc[:, 2] < top + 0.06) & (fc[:, 2] > bot - 0.06) & (np.abs(fc[:, 0]) < 0.3))
    subdivide(ob, 2)
    co = get_verts(ob)
    A = geo.adjacency(faces_of(ob), len(co))
    edge = geo.boundary_mask(faces_of(ob), len(co))
    co = geo.drape(co, A, ctx.body_bvh, 0.012, iters=6, lam=0.5, pin=edge)
    for s, sx in (("l", 1), ("r", -1)):
        ch = leg_chain(ctx, s)
        sp, c, d, _ = geo.chain_param(co, ch)
        z = co[:, 2]
        side = smoothstep(-0.02, 0.03, co[:, 0] * sx)
        knee_z = ch[1][2]
        # balloon: grows to the knee, peaks over the calf, tucks into the wraps with an overhang
        ease = (0.010 + 0.05 * smoothstep(top, knee_z + 0.1, z) + 0.025 * smoothstep(knee_z + 0.15, knee_z - 0.05, z))
        ease *= smoothstep(bot + 0.02, bot + 0.09, z) * 0.95 + 0.05
        r = co - c
        r[:, 2] *= 0.3
        rn = r / np.maximum(np.linalg.norm(r, axis=1, keepdims=True), 1e-6)
        inner = np.clip(-rn[:, 0] * sx, 0, 1)
        ease *= 1 - 0.55 * inner * smoothstep(knee_z - 0.05, top - 0.15, z)
        co = co + rn * (ease * side)[:, None]
        # blousing fabric droops over the wrap top
        droop = 0.03 * smoothstep(bot + 0.14, bot + 0.05, z) * smoothstep(bot + 0.01, bot + 0.05, z)
        co[:, 2] -= droop * side
    co = geo.smooth(co, A, 0.35, 3)
    disp = np.zeros(len(co))
    for k, (s, sx) in enumerate((("l", 1), ("r", -1))):
        ch = leg_chain(ctx, s)
        sv, phi, rad, rn = folds.limb_coords(co, ch)
        side = smoothstep(-0.02, 0.03, co[:, 0] * sx)
        z = co[:, 2]
        knee_z = ch[1][2]
        twist = folds.pipes(phi * 0.11 + sv * 0.55 * (1 if k else -1), sv, 0.09, 0.16, 71 + k, 0.8)
        hang = folds.pipes(phi * 0.11, sv, 0.10, 0.35, 73 + k, 0.8)
        bunch = folds.pipes(sv, phi * 0.11, 0.055, 0.16, 75 + k, 0.6)
        a_tw = 0.011 * smoothstep(top - 0.1, knee_z + 0.1, z) * smoothstep(bot + 0.02, bot + 0.12, z)
        a_hg = 0.008 * smoothstep(top, knee_z, z)
        a_bu = 0.010 * smoothstep(bot + 0.2, bot + 0.07, z) * smoothstep(bot + 0.02, bot + 0.06, z)
        disp += side * (twist * a_tw + hang * a_hg + bunch * a_bu)
    co = folds.apply(co, faces_of(ob), disp)
    co = geo.push_out(co, ctx.body_bvh, 0.008)
    # tuck the cuff inside the wraps: pull the last few cm back onto the calf
    loc, nrm, _ = geo.nearest(co, ctx.body_bvh)
    tuck = smoothstep(bot + 0.035, bot + 0.005, co[:, 2])[:, None]
    co = co * (1 - tuck) + (loc + nrm * 0.0035) * tuck
    set_verts(ob, co)
    bisect(ob, (0, 0, top), (0, 0, 1), clear_outer=True)
    bisect(ob, (0, 0, bot), (0, 0, -1), clear_outer=True)
    return ob


def leg_wraps(ctx):
    ob = extract(ctx, "LegWraps", ["helper-tights"],
                 lambda fc, fv: (fc[:, 2] < WRAP_TOP + 0.05) & (fc[:, 2] > WRAP_BOT - 0.04))
    subdivide(ob, 2)
    co = get_verts(ob)
    A = geo.adjacency(faces_of(ob), len(co))
    co = geo.drape(co, A, ctx.body_bvh, 0.006, iters=6, lam=0.5)
    # overlapping spiral bandage layers: sawtooth displacement along a helix
    for s, sx in (("l", 1), ("r", -1)):
        ch = leg_chain(ctx, s)
        sp, c, d, _ = geo.chain_param(co, ch)
        m = co[:, 0] * sx > 0
        rel = co - c
        th = np.arctan2(rel[:, 1], rel[:, 0] * sx)
        pitch = 0.042
        u = (co[:, 2] + th / (2 * np.pi) * pitch) / pitch
        fr = u - np.floor(u)
        saw = 0.0022 * fr + 0.0008 * np.exp(-((fr - 0.97) / 0.03) ** 2)
        nrm = rel / np.maximum(np.linalg.norm(rel, axis=1, keepdims=True), 1e-6)
        co[m] += nrm[m] * saw[m, None]
    set_verts(ob, co)
    bisect(ob, (0, 0, WRAP_TOP), (0, 0, 1), clear_outer=True)
    bisect(ob, (0, 0, WRAP_BOT), (0, 0, -1), clear_outer=True)
    return ob


def wrap_straps(ctx, wraps, turns=2.0, width=0.013, lift=0.0025):
    """Black straps criss-crossing the wraps (two opposite helices + ties top and bottom)."""
    bvh = geo.bvh_from_object(wraps)
    out = []
    from mathutils import Vector
    for s, sx in (("l", 1), ("r", -1)):
        ch = leg_chain(ctx, s)
        z0, z1 = WRAP_BOT + 0.03, WRAP_TOP - 0.035

        def axis_at(z):
            a, b = ch[1], ch[2]
            t = (z - a[2]) / (b[2] - a[2])
            return a + (b - a) * t

        def surf(z, th):
            c = axis_at(z)
            d = np.array([math.cos(th) * sx, math.sin(th), 0.0])
            hit = bvh.ray_cast(Vector(c + d * 0.11), Vector(-d))
            if hit[0] is None:
                return c + d * 0.06, d
            return np.array(hit[0]), np.array(hit[1])

        paths = []
        for hand in (1, -1):
            n = 90
            pts = []
            for k in range(n):
                t = k / (n - 1)
                z = z0 + (z1 - z0) * t
                th = -np.pi / 2 + hand * 2 * np.pi * turns * t
                pts.append((z, th))
            paths.append(pts)
        for z in (z1 + 0.012, z0 - 0.012):        # horizontal ties
            paths.append([(z, -np.pi / 2 + 2 * np.pi * k / 60) for k in range(61)])
        for pi, path in enumerate(paths):
            P, N = zip(*[surf(z, th) for z, th in path])
            P, N = np.array(P), np.array(N)
            rings = []
            for k in range(len(P)):
                t = P[min(k + 1, len(P) - 1)] - P[max(k - 1, 0)]
                t /= np.linalg.norm(t)
                b = np.cross(N[k], t)
                b /= np.linalg.norm(b)
                up = lift + 0.0012 * pi           # stack crossings so straps do not z-fight
                rings.append([P[k] + N[k] * (up - 0.0008) - b * width / 2,
                              P[k] + N[k] * (up + 0.0008) - b * width / 4,
                              P[k] + N[k] * (up + 0.0012),
                              P[k] + N[k] * (up + 0.0008) + b * width / 4,
                              P[k] + N[k] * (up - 0.0008) + b * width / 2])
            out.append(loft(f"Strap_{s}{pi}", np.array(rings), closed=False))
    return out


# --------------------------------------------------------------------------------------
# canvas shoes

SHOE_TOP = 0.112


def _star_outline(xy, n=72, pad=0.004):
    c = xy.mean(0)
    rel = xy - c
    a = np.arctan2(rel[:, 1], rel[:, 0])
    r = np.linalg.norm(rel, axis=1)
    bins = np.linspace(-np.pi, np.pi, n + 1)
    out = np.zeros(n)
    for i in range(n):
        m = (a >= bins[i]) & (a < bins[i + 1])
        out[i] = r[m].max() if m.any() else np.nan
    ok = ~np.isnan(out)
    idx = np.arange(n)
    out = np.interp(idx, idx[ok], out[ok], period=n)
    for _ in range(3):
        out = 0.5 * out + 0.25 * (np.roll(out, 1) + np.roll(out, -1))
    mid = 0.5 * (bins[:-1] + bins[1:])
    return c, mid, out + pad


def shoes(ctx):
    """Low canvas sneakers: lofted upper sized from the foot, rubber sole, laces."""
    from .body import SOLE
    from mathutils import Vector
    parts = []
    co_b = get_verts(ctx.body)
    ids = np.zeros(len(co_b), dtype=np.int32)
    ctx.body.data.attributes["mh_id"].data.foreach_get("value", ids)
    for s, sx in (("l", 1), ("r", -1)):
        ankle = ctx.j(f"foot_{s}.head")
        ball = ctx.j(f"ball_{s}.head")
        fwd = ball - ankle
        fwd[2] = 0
        fwd /= np.linalg.norm(fwd)
        lat = np.cross([0, 0, 1.0], fwd)
        foot = co_b[(co_b[:, 2] < 0.13) & (co_b[:, 0] * sx > 0.04)]
        al = (foot - ankle) @ fwd
        la = (foot - ankle) @ lat
        a0, a1 = al.min() - 0.006, al.max() + 0.018
        n_st, M = 30, 32
        st = np.linspace(a0, a1, n_st)
        rings = []
        prof = []
        for k, a in enumerate(st):
            t = (a - a0) / (a1 - a0)
            m = np.abs(al - np.clip(a, al.min() + 0.004, al.max() - 0.004)) < 0.009
            lo, hi = la[m].min(), la[m].max()
            top = min(foot[m][:, 2].max(), 0.125)
            clr = 0.006
            w = (hi - lo) / 2 + clr
            cl = (hi + lo) / 2
            # toe cap: rounded and slightly lower; heel: rounded counter
            toe = smoothstep(0.80, 1.0, t)
            heel = smoothstep(0.08, 0.0, t)
            w *= np.sqrt(np.clip(1 - toe ** 2.2, 0.02, 1)) * np.sqrt(np.clip(1 - heel ** 2.2, 0.02, 1))
            h = (top - SOLE) + clr * 0.6
            h = h * (1 - 0.45 * toe) * (1 - 0.25 * heel)
            h = max(h, 0.012)
            ring = []
            for j in range(M):
                th = 2 * np.pi * j / M
                c, sn = np.cos(th), np.sin(th)
                e = 2.6
                xx = np.sign(c) * abs(c) ** (2 / e) * w
                zz = np.sign(sn) * abs(sn) ** (2 / e)
                zz = SOLE + 0.004 + (zz * 0.5 + 0.5) * h
                ring.append(ankle + fwd * a + lat * (cl + xx) + [0, 0, zz - ankle[2]])
            rings.append(ring)
            prof.append((a, cl - w, cl + w))
        rings = np.array(rings)
        up = loft(f"Shoe_{s}", rings, closed=True)
        _close_ends(up)
        subdivide(up, 1)
        bisect(up, (0, 0, SHOE_TOP), (0, 0, 1), clear_outer=True)
        set_verts(up, geo.smooth(get_verts(up), geo.adjacency(faces_of(up), len(up.data.vertices)), 0.2, 2,
                                 mask=(~geo.boundary_mask(faces_of(up), len(up.data.vertices))).astype(float)))
        parts.append(up)
        # rubber sole: outline from the upper's footprint, rounded wall rising over the upper
        outline = []
        for a, lo, hi in prof:
            outline.append(ankle + fwd * a + lat * (lo - 0.004))
        for a, lo, hi in prof[::-1]:
            outline.append(ankle + fwd * a + lat * (hi + 0.004))
        outline = np.array(outline)
        outline[:, 2] = 0
        c2 = outline.mean(0)
        for _ in range(4):
            outline = 0.5 * outline + 0.25 * (np.roll(outline, 1, 0) + np.roll(outline, -1, 0))
        srings = []
        for zz, d in ((0.0, -0.003), (0.0015, 0.0), (SOLE * 0.5, 0.0012), (SOLE + 0.006, 0.0008),
                      (SOLE + 0.011, -0.0012), (SOLE + 0.0125, -0.004)):
            rr = outline + (outline - c2) / np.linalg.norm(outline - c2, axis=1, keepdims=True) * d
            rr[:, 2] = zz
            srings.append(rr)
        sole = loft(f"Sole_{s}", np.array(srings), closed=True)
        _cap(sole, 0)
        parts.append(sole)
        # laces across the instep
        bvh = geo.bvh_from_object(up)
        for k in range(5):
            a = st[0] + (st[-1] - st[0]) * (0.34 + 0.065 * k)
            rows = []
            for u in np.linspace(-0.018, 0.018, 7):
                q0 = ankle + fwd * a + lat * u
                q0[2] = 0.2
                hh = bvh.ray_cast(Vector(q0), Vector((0, 0, -1)))
                if hh[0] is None:
                    rows = []
                    break
                q = np.array(hh[0])
                rows.append([q + fwd * 0.003 + [0, 0, 0.001], q + [0, 0, 0.0028], q - fwd * 0.003 + [0, 0, 0.001]])
            if rows:
                parts.append(loft(f"Lace_{s}{k}", np.array(rows), closed=False))
    # skin weights from the nearest foot vertices
    from scipy.spatial import cKDTree
    tree = cKDTree(co_b)
    for p in parts:
        pc = get_verts(p)
        d, i = tree.query(pc, k=6)
        w = 1 / np.maximum(d, 1e-4) ** 2
        w /= w.sum(1, keepdims=True)
        W = (ctx.Wfull[ids[i]] * w[:, :, None]).sum(1)
        write_groups(p, limit(W), ctx.bone_names)
    return parts


def _close_ends(ob):
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    edges = [e for e in bm.edges if len(e.link_faces) == 1]
    bmesh.ops.holes_fill(bm, edges=edges, sides=0)
    bmesh.ops.triangulate(bm, faces=[f for f in bm.faces if len(f.verts) > 4])
    bm.to_mesh(ob.data)
    bm.free()


def _cap(ob, ring_index):
    """Fill the open ring at `ring_index` (of a loft) with a single n-gon."""
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(ob.data)
    bm.verts.ensure_lookup_table()
    edges = [e for e in bm.edges if len(e.link_faces) == 1]
    zs = [e.verts[0].co.z for e in edges]
    zmin = min(zs)
    loop = [e for e in edges if abs(e.verts[0].co.z - zmin) < 1e-5 and abs(e.verts[1].co.z - zmin) < 1e-5]
    bmesh.ops.holes_fill(bm, edges=loop, sides=0)
    bmesh.ops.triangulate(bm, faces=[f for f in bm.faces if len(f.verts) > 4])
    bm.to_mesh(ob.data)
    bm.free()
