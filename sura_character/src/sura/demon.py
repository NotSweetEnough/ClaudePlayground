"""The Sura's demonic left arm: corrupted forearm/hand mask, sinew sculpt, claws and bone spikes."""
import math

import numpy as np

from . import folds, geo
from .butil import get_verts, set_verts, smoothstep
from .clothing import loft


def _bone(arm, name):
    b = arm.data.bones[name]
    m = np.array(b.matrix_local)
    return np.array(b.head_local), np.array(b.tail_local), m[:3, 0], m[:3, 1], m[:3, 2]


def arm_param(arm, co):
    """Arc length along shoulder -> elbow -> wrist -> middle fingertip of the left arm."""
    pts = [np.array(arm.data.bones["upperarm_l"].head_local), np.array(arm.data.bones["upperarm_l"].tail_local),
           np.array(arm.data.bones["lowerarm_l"].tail_local), np.array(arm.data.bones["middle_01_l"].head_local),
           np.array(arm.data.bones["middle_03_l"].tail_local)]
    s, c, d, seg = geo.chain_param(co, pts)
    L_up = np.linalg.norm(pts[1] - pts[0])
    return s, d, L_up


def demon_mask(arm, co):
    """1 on the demonic forearm/hand, ragged fade up the upper arm. Left side only."""
    s, d, L_up = arm_param(arm, co)
    n = geo.value_noise3(co, 22.0, seed=5, octaves=3)
    edge = L_up - 0.02 + 0.05 * n                       # ragged corruption front just above the elbow
    m = smoothstep(edge - 0.06, edge + 0.04, s)
    m *= (co[:, 0] > 0.12) & (d < 0.12)
    thumb = [np.array(arm.data.bones[f"thumb_0{k}_l"].head_local) for k in (1, 2, 3)]
    thumb.append(np.array(arm.data.bones["thumb_03_l"].tail_local))
    _, _, dt, _ = geo.chain_param(co, thumb)
    return np.maximum(m, ((dt < 0.03) & (co[:, 0] > 0.12)).astype(float))


def sculpt(body, arm):
    """Sinewy, knotted forearm; swollen knuckles; bony finger joints."""
    co = get_verts(body)
    faces = [tuple(p.vertices) for p in body.data.polygons]
    m = demon_mask(arm, co)
    h, t, X, Y, Z = _bone(arm, "lowerarm_l")
    sv, phi, rad, rn = folds.limb_coords(co, [h, t])
    cords = folds.pipes(phi * 0.04, sv, 0.018, 0.10, 91, 0.7)
    disp = m * cords * 0.0025 * smoothstep(0.02, 0.08, sv)
    for f in ("index", "middle", "ring", "pinky"):
        for j, w, amp in ((f"{f}_01_l", "head", 0.0032), (f"{f}_02_l", "head", 0.0018), (f"{f}_03_l", "head", 0.0012)):
            p = np.array(arm.data.bones[j].head_local)
            disp += m * amp * np.exp(-np.sum((co - p) ** 2, 1) / (2 * 0.009 ** 2))
    p = np.array(arm.data.bones["lowerarm_l"].tail_local)   # wrist bones
    disp += m * 0.002 * np.exp(-np.sum((co - p) ** 2, 1) / (2 * 0.02 ** 2))
    nrm = geo.vertex_normals(co, faces)
    set_verts(body, co + nrm * disp[:, None])
    return m


def _horn(name, base, direction, bend_dir, length, r0, r1=0.0, bend=0.35, segs=14, sides=10, flat=1.0, keel=None, taper=0.85):
    """Curved tapered horn / claw as a lofted tube with a closed tip."""
    direction = direction / np.linalg.norm(direction)
    bend_dir = bend_dir - (bend_dir @ direction) * direction
    bend_dir /= np.linalg.norm(bend_dir)
    rings = []
    pts = []
    for k in range(segs + 1):
        t = k / segs
        p = base + direction * length * t + bend_dir * length * bend * t * t
        pts.append(p)
    pts = np.array(pts)
    for k in range(segs + 1):
        t = k / segs
        tan = pts[min(k + 1, segs)] - pts[max(k - 1, 0)]
        tan /= np.linalg.norm(tan)
        b = bend_dir - (bend_dir @ tan) * tan
        b /= np.linalg.norm(b)
        sd = np.cross(tan, b)
        r = r0 + (r1 - r0) * t
        r *= (1 - t) ** taper if k < segs else 0.0
        ring = []
        for a in np.linspace(0, 2 * np.pi, sides, endpoint=False):
            ca, sa = math.cos(a), math.sin(a)
            kk = 1.0
            if keel is not None and sa > 0:            # sharpened edge on the inner (palm) side
                kk = 1 + keel * sa ** 4
            ring.append(pts[k] + sd * ca * r + b * sa * r * flat * kk)
        rings.append(ring)
    ob = loft(name, np.array(rings), closed=True)
    ob.data.shade_smooth()
    return ob


def claws(arm):
    out = []
    for f, L, r in (("thumb", 0.040, 0.0085), ("index", 0.048, 0.0082), ("middle", 0.052, 0.0085),
                    ("ring", 0.048, 0.0080), ("pinky", 0.040, 0.0072)):
        h, t, X, Y, Z = _bone(arm, f"{f}_03_l")
        seg = np.linalg.norm(t - h)
        base = h + Y * seg * 0.25 - Z * r * 0.55               # sits on the nail bed (dorsal side)
        direction = Y * 0.92 + Z * 0.12
        out.append(_horn(f"Claw_{f}", base, direction, Z, L + seg * 0.6, r, 0.0, bend=0.5,
                         segs=16, sides=12, flat=0.8, keel=0.5, taper=0.5))
    return out


def spikes(arm, body):
    """Bone spikes along the ulna ridge and spurs on the knuckles."""
    from mathutils import Vector
    bvh = geo.bvh_from_object(body)
    out = []
    h, t, X, Y, Z = _bone(arm, "lowerarm_l")
    pinky = np.array(arm.data.bones["pinky_metacarpal_l"].head_local) - np.array(arm.data.bones["index_metacarpal_l"].head_local)
    pinky -= (pinky @ Y) * Y
    pinky /= np.linalg.norm(pinky)
    dorsal = -np.mean([_bone(arm, f"{f}_01_l")[4] for f in ("index", "middle", "ring", "pinky")], axis=0)
    dorsal = dorsal - (dorsal @ Y) * Y
    dorsal /= np.linalg.norm(dorsal)
    ridge = pinky * 0.55 + dorsal * 0.8
    ridge /= np.linalg.norm(ridge)
    L = np.linalg.norm(t - h)
    for k, (f, length, r) in enumerate(((0.40, 0.042, 0.0105), (0.56, 0.056, 0.0125), (0.72, 0.044, 0.0105), (0.87, 0.026, 0.0075))):
        c = h + Y * L * f
        hit = bvh.ray_cast(Vector(c + ridge * 0.12), Vector(-ridge))
        if hit[0] is None:
            continue
        base = np.array(hit[0]) - ridge * 0.004
        d = ridge * 0.75 - Y * 0.65
        out.append(_horn(f"Spike_arm{k}", base, d, -Y, length, r, 0.0, bend=0.25, segs=10, sides=10))
    for k, f in enumerate(("index", "middle", "ring", "pinky")):
        p = np.array(arm.data.bones[f"{f}_01_l"].head_local)
        _, _, fX, fY, fZ = _bone(arm, f"{f}_01_l")
        hit = bvh.ray_cast(Vector(p - fZ * 0.06), Vector(fZ))
        if hit[0] is None:
            continue
        base = np.array(hit[0]) + fZ * 0.002
        out.append(_horn(f"Spur_{f}", base, -fZ * 0.8 + fY * 0.5, fY, 0.015 - 0.0015 * abs(k - 1.5), 0.0052, 0.0,
                         bend=0.3, segs=6, sides=8))
    return out
