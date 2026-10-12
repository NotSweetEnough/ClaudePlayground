"""Numpy / mathutils geometry helpers used for garment shaping."""
import numpy as np
import scipy.sparse as sp
from mathutils import Vector
from mathutils.bvhtree import BVHTree


def adjacency(faces, n):
    rows, cols = [], []
    for f in faces:
        k = len(f)
        for i in range(k):
            a, b = f[i], f[(i + 1) % k]
            rows += [a, b]
            cols += [b, a]
    A = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
    A.data[:] = 1.0
    return A


def boundary_mask(faces, n):
    from collections import Counter
    c = Counter()
    for f in faces:
        k = len(f)
        for i in range(k):
            a, b = f[i], f[(i + 1) % k]
            c[(min(a, b), max(a, b))] += 1
    m = np.zeros(n, dtype=bool)
    for (a, b), cnt in c.items():
        if cnt == 1:
            m[a] = m[b] = True
    return m


def smooth(co, A, lam=0.5, iters=1, mask=None):
    deg = np.asarray(A.sum(1)).ravel()
    deg[deg == 0] = 1
    for _ in range(iters):
        avg = (A @ co) / deg[:, None]
        d = (avg - co) * lam
        if mask is not None:
            d *= mask[:, None]
        co = co + d
    return co


def smooth_field(f, A, iters=1, lam=0.5):
    return smooth(f.reshape(len(f), -1).astype(float), A, lam, iters).reshape(f.shape)


def bvh_from_object(ob):
    me = ob.data
    co = [ob.matrix_world @ v.co for v in me.vertices]
    polys = [tuple(p.vertices) for p in me.polygons]
    return BVHTree.FromPolygons(co, polys)


def bvh_from_arrays(co, faces):
    return BVHTree.FromPolygons([Vector(c) for c in co], [tuple(f) for f in faces])


def push_out(co, bvh, offset, strength=1.0):
    """Ensure every point is at least `offset` (scalar or per-vertex) outside the surface."""
    off = np.broadcast_to(np.asarray(offset, dtype=float), (len(co),))
    out = co.copy()
    for i, p in enumerate(co):
        loc, nrm, _, dist = bvh.find_nearest(Vector(p))
        if loc is None:
            continue
        d = Vector(p) - loc
        signed = d.dot(nrm)
        if signed < off[i]:
            target = np.array(loc + nrm * off[i])
            out[i] = p + (target - p) * strength
    return out


def nearest(co, bvh):
    locs, nrms = np.zeros_like(co), np.zeros_like(co)
    dist = np.zeros(len(co))
    for i, p in enumerate(co):
        loc, nrm, _, d = bvh.find_nearest(Vector(p))
        locs[i], nrms[i], dist[i] = loc, nrm, d
    return locs, nrms, dist


def drape(co, A, bvh, offset, iters=20, lam=0.6, pin=None, gravity=0.0):
    """Pseudo cloth: alternately smooth (cloth bridges concavities) and push outside the body."""
    mask = None if pin is None else (~pin).astype(float)
    for _ in range(iters):
        co = smooth(co, A, lam, 1, mask)
        if gravity:
            g = np.zeros_like(co)
            g[:, 2] = -gravity
            if mask is not None:
                g *= mask[:, None]
            co = co + g
        co = push_out(co, bvh, offset)
    return co


def vertex_normals(co, faces):
    n = np.zeros_like(co)
    for f in faces:
        p = co[list(f)]
        k = len(f)
        for i in range(k):
            a, b, c = p[i - 1], p[i], p[(i + 1) % k]
            n[f[i]] += np.cross(c - b, a - b)
    l = np.linalg.norm(n, axis=1, keepdims=True)
    l[l == 0] = 1
    return n / l


def seg_project(p, a, b):
    """Project points p (N,3) on segment a-b: returns (t in [0,1], closest point, distance)."""
    ab = b - a
    t = np.clip(((p - a) @ ab) / (ab @ ab), 0, 1)
    c = a + t[:, None] * ab
    return t, c, np.linalg.norm(p - c, axis=1)


def chain_param(p, pts):
    """Arc-length parameter of points projected onto the closest segment of a polyline."""
    best_d = np.full(len(p), np.inf)
    s_out = np.zeros(len(p))
    c_out = np.zeros_like(p)
    seg_i = np.zeros(len(p), dtype=int)
    acc = 0.0
    for i in range(len(pts) - 1):
        a, b = np.asarray(pts[i]), np.asarray(pts[i + 1])
        t, c, d = seg_project(p, a, b)
        L = np.linalg.norm(b - a)
        better = d < best_d
        best_d[better] = d[better]
        s_out[better] = acc + t[better] * L
        c_out[better] = c[better]
        seg_i[better] = i
        acc += L
    return s_out, c_out, best_d, seg_i


def value_noise3(p, freq, seed=0, octaves=3):
    """Smooth 3D value noise in [-1, 1] (numpy, deterministic)."""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(256)
    perm = np.concatenate([perm, perm])
    vals = rng.uniform(-1, 1, 256)
    total = np.zeros(len(p))
    amp, norm = 1.0, 0.0
    f = freq
    for o in range(octaves):
        q = p * f + o * 17.31
        i0 = np.floor(q).astype(int)
        fr = q - i0
        w = fr * fr * (3 - 2 * fr)
        res = 0.0
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    h = perm[(perm[(perm[(i0[:, 0] + dx) & 255] + i0[:, 1] + dy) & 255] + i0[:, 2] + dz) & 255]
                    wx = w[:, 0] if dx else 1 - w[:, 0]
                    wy = w[:, 1] if dy else 1 - w[:, 1]
                    wz = w[:, 2] if dz else 1 - w[:, 2]
                    res = res + vals[h] * wx * wy * wz
        total += amp * res
        norm += amp
        amp *= 0.5
        f *= 2.0
    return total / norm
