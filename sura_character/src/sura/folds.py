"""Procedural cloth folds (geometry level). Fine wrinkles go to the baked normal maps."""
import numpy as np

from . import geo


def aniso_noise(u, v, fu, fv, seed, octaves=3):
    p = np.stack([u * fu, v * fv, np.full_like(u, seed * 1.618)], 1)
    return geo.value_noise3(p, 1.0, seed, octaves)


def fold_shape(n, crease=0.45):
    """Soft rounded crests mixed with pinched valleys."""
    n = np.clip(n * 1.6, -1, 1)
    soft = n
    pinched = 1 - 2 * (1 - np.abs(n)) ** 0.5 * (n < 0) - (n >= 0) * (1 - n) ** 2
    return (1 - crease) * soft + crease * pinched * 0.6


def limb_coords(co, chain):
    s, c, d, _ = geo.chain_param(co, chain)
    a = np.asarray(chain[0])
    b = np.asarray(chain[-1])
    ax = (b - a) / np.linalg.norm(b - a)
    ref = np.cross(ax, [0, 0, 1.0])
    if np.linalg.norm(ref) < 1e-3:
        ref = np.cross(ax, [0, 1.0, 0])
    ref /= np.linalg.norm(ref)
    ref2 = np.cross(ax, ref)
    r = co - c
    phi = np.arctan2(r @ ref2, r @ ref)
    rad = np.linalg.norm(r, axis=1)
    return s, phi, rad, r / np.maximum(rad[:, None], 1e-6)


def torso_coords(co, yc=0.0):
    phi = np.arctan2(co[:, 1] - yc, co[:, 0])
    rad = np.hypot(co[:, 0], co[:, 1] - yc)
    return phi, rad


def apply(co, faces, disp, body_bvh=None, min_off=0.003, smooth_A=None):
    nrm = geo.vertex_normals(co, faces)
    out = co + nrm * disp[:, None]
    if smooth_A is not None:
        out = geo.smooth(out, smooth_A, 0.25, 1)
    if body_bvh is not None:
        out = geo.push_out(out, body_bvh, min_off)
    return out


def pipes(u, v, wavelength, along, seed, warp=1.2):
    """Long hanging 'pipe' folds: phase-warped sine with broad crests and pinched valleys.

    u: coordinate across the folds (m), v: along the folds (m)."""
    w = aniso_noise(u, v, 1 / (wavelength * 2.5), 1 / along, seed, 2)
    am = 0.45 + 0.55 * np.clip(aniso_noise(u, v, 1 / (wavelength * 1.7), 1 / (along * 0.6), seed + 101, 2) * 1.8 + 0.5, 0, 1)
    g = np.sin(2 * np.pi * u / wavelength + warp * 2 * np.pi * w)
    shaped = 1 - 2 * ((1 - g) / 2) ** 1.6
    return shaped * am
